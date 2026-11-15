"""
Stage 0: 원본 프롬프트 확정 (MC/language bias 완화 재샘플링 + 최종 조립).

이 파일은 지금까지 별도로 진행했던 3개의 소스별 재작업을 "Stage 0"으로 묶어서
문서화하고, 최종 조립(assembly)만 실행하는 스크립트다. LLM 호출이 필요한 무거운
단계(아래 A/B)는 이미 완료되어 결과 파일이 남아있으므로 여기서 다시 돌리지 않고
그 산출물을 그대로 읽어 합친다. 재현이 필요하면 각 하위 스크립트를 직접 실행한다.

  A) MMLU -- MC(4지선다) 형식 자체가 서프면-폼 편향이라 open-style로 전면 재작성.
     Open-LLM-Benchmark(OSQ-bench) 소스에서 subject 비율 유지 층화추출로 700건 확정.
       재현: pipeline/stage0/mmlu_openstyle/sample_mmlu_openstyle_700.py
             (HF datasets만 사용, LLM 호출 없음 -- 그대로 재실행 가능)
       산출물: data/stage0_inputs/mmlu_openstyle_700.json (700건)

  B) ScienceQA -- 2단계 필터링으로 "이미지가 실제로 필요한" 문항만 선별.
       Step1 (language bias, blind test): 이미지 없이 선지만 주고 gpt-5.2가 3회 반복
         답변 -> 3회 중 과반 정답이면 language-biased로 판정, 탈락.
       Step2 (선지 의존도): 선지 없이 이미지+질문만으로 유일하게 답 가능한지 판정.
       전체 2,017건 -> Step1 생존 653 -> Step2 생존 409.
       재현: pipeline/stage0/scienceqa_filtering/{blind_test_full.py, run_gpt_audit_v5_full.py}
             (원본 2,017건 pool과 이미지는 이 저장소에 없음 -- 루트 README.md "Stage 0 sub-pipelines" 참고)
       산출물: data/stage0_inputs/scienceqa_final_409.json (409건)
       리포트: /nas/home/ysk1207/nips/after-rebuttar/scienceqa_filtering_report.md
               (이 기기 로컬 경로, 레포에는 없음 -- 필요시 별도 공유)

  C) VQAv2 -- ScienceQA와 동일한 blind test(language bias)만 적용(선지가 없는
     open-ended라 Step2는 해당 없음). 이미지 없이 질문만 주고 3회 반복 답변 후
     공식 VQA 정규화(관사 제거/숫자단어 변환 등) 거쳐 10명 중 3명 이상 동의한 답과
     일치하면 정답 -> 과반 정답이면 탈락.
       파일럿 200(층화, yes/no 1/3 캡) -> 생존율 확인 -> 최종 2,400(단순 무작위) 전수
       -> 생존 1,541 -> 그중 991건 무작위 확정(SEED=42).
       재현: pipeline/stage0/vqav2_filtering/{prepare_pool.py, blind_test_full.py,
             finalize_survivors.py, select_final_991.py}
             (공식 VQAv2 val2014 questions/annotations는 이 저장소에 없음 -- README 참고)
       산출물: data/stage0_inputs/vqav2_final_ti2t_991.json (991건)
       리포트: /nas/home/ysk1207/nips/after-rebuttar/vqav2_filtering_report.md
               (이 기기 로컬 경로, 레포에는 없음 -- 필요시 별도 공유)

  D) 나머지 5개 소스(gsm8k/geneval/dpg_bench/MagicBrush/ImgEditBench)는 원래
     28,410-variant 확장판(comi_bench_full_refined.json)에서 그대로 가져온다.
     단 gsm8k는 이 확장판에 gold_answer가 아예 없었어서 여기서 새로 복구한다
     (openai/gsm8k train+test 전체와 원문 텍스트 완전일치, 700/700 매칭).

  700(A) + 409(B) + 991(C) + 700+553+1065+527+737(D, 그대로) = 5,682
  (ScienceQA 부족분 291을 VQAv2가 정확히 메꿔서 총량이 원래 base 5,682와 동일하게 유지됨)

이 스크립트가 실제로 실행하는 것은 위 A/B/C/D의 산출물을 공통 스키마로 정규화해서
합치는 "조립"뿐이다. 출력은 두 가지:
  1) rich json (gold_answer/image_path/metadata 포함, 채점·추론 단계에서 필요)
  2) stage1_judge.py가 바로 읽는 samples.jsonl 포맷(id/prompt/task/source/note).
     jsonl 쪽에도 rich json의 필드를 전부 실어 보낸다 -- stage1/stage2는
     `{**sample, ...}`로 원본 필드를 그대로 보존하며 지나가므로, 원본 필드가
     이후 단계까지 안 끊기고 들고 다녀진다(image_path로 실제 이미지를 참조해야
     하는 TI2T/TI2I 소스 때문에 특히 중요).

사용법
------
    python stage0_prepare_originals.py
    python stage0_prepare_originals.py --out-json /home/ysk1207/comi/data/stage0_originals.json \\
                                        --out-jsonl /home/ysk1207/comi/data/stage0_originals.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COMI_FULL = DATA_DIR / "stage0_inputs" / "comi_bench_full_refined.json"
MMLU_OPENSTYLE = DATA_DIR / "stage0_inputs" / "mmlu_openstyle_700.json"
SCIENCEQA_409 = DATA_DIR / "stage0_inputs" / "scienceqa_final_409.json"
VQAV2_991 = DATA_DIR / "stage0_inputs" / "vqav2_final_ti2t_991.json"

DEFAULT_OUT_JSON = str(DATA_DIR / "comi_bench_final_original_5682.json")
DEFAULT_OUT_JSONL = str(DATA_DIR / "stage0_originals.jsonl")

UNCHANGED_SOURCES = ["gsm8k", "geneval", "dpg_bench", "MagicBrush", "ImgEditBench"]

# stage1_judge.py의 per_note 리포트에서 어떤 하위 단계 산출물인지 바로 구분되도록
NOTE_BY_SOURCE_GROUP = {
    "mmlu": "mc_openstyle_rewrite",           # A
    "ScienceQA": "language_bias_filtered",    # B
    "VQAv2": "language_bias_filtered",        # C
}
DEFAULT_NOTE = "unchanged_original"           # D


def recover_gsm8k_gold(gsm_items: list[dict]) -> list[dict]:
    import datasets  # HF hub에서 받음 -- 로컬 캐시 스냅샷 경로에 의존하지 않음

    gsm = datasets.load_dataset("openai/gsm8k", "main")

    def extract_final(a: str) -> str | None:
        m = re.search(r"####\s*(.+)", a)
        return m.group(1).strip().replace(",", "") if m else None

    by_q = {}
    for split in ("train", "test"):
        for row in gsm[split]:
            by_q[row["question"].strip()] = extract_final(row["answer"])

    matched = 0
    for r in gsm_items:
        gold = by_q.get(r["original_question"].strip())
        r["gold_answer"] = gold
        matched += int(gold is not None)
    print(f"  [D] gsm8k gold_answer 복구: {matched}/{len(gsm_items)}")
    return gsm_items


def clean_unchanged(items: list[dict]) -> list[dict]:
    """D: 기존 5개 소스 -- 이전 refine_pipeline 산출물(expected/refine_comment/
    refined_variants 등) 제거하고 공통 스키마로 정리."""
    return [{
        "task_id": r["task_id"],
        "source": r["source"],
        "original_question": r["original_question"],
        "task": r["task"],
        "gold_answer": r.get("gold_answer"),
        "gold_modality": r["gold_modality"],
        "image_path": r["image_path"],
        "metadata": {},
    } for r in items]


def normalize_mmlu_openstyle(items: list[dict]) -> list[dict]:
    """A"""
    return [{
        "task_id": r["task_id"],
        "source": "mmlu",
        "original_question": r["original_question"],
        "task": "t2t",
        "gold_answer": r["gold_answer"],
        "gold_modality": "text",
        "image_path": None,
        "metadata": {
            "gold_answer_key": r.get("gold_answer_key"),
            "mmlu_subject": r.get("mmlu_subject"),
            "first_filter": r.get("first_filter"),
            "second_filter": r.get("second_filter"),
        },
    } for r in items]


def normalize_scienceqa_409(items: list[dict]) -> list[dict]:
    """B"""
    out = []
    for r in items:
        pid = r["pid"]
        out.append({
            "task_id": f"scienceqa_filtered_{pid}",
            "source": "ScienceQA",
            "original_question": r["prompt"],  # 선지(A. B. ...) 포함, 기존 컨벤션과 동일
            "task": "ti2t",
            "gold_answer": r["answer"],
            "gold_modality": "text",
            "image_path": r["image_path"],
            "metadata": {
                "pid": pid,
                "hint": r.get("hint"),
                "topic": r.get("topic"),
                "category": r.get("category"),
                "subject": r.get("subject"),
                "gold_choices": r.get("choices"),
                "blind_correct_count": r.get("blind_correct_count"),
                "verdict_v5": r.get("verdict_v5"),
                "reasoning_v5": r.get("reasoning_v5"),
                "prompt_ko": r.get("prompt_ko"),
            },
        })
    return out


def normalize_vqav2_991(items: list[dict]) -> list[dict]:
    """C -- 이미 공통 스키마라 필드만 추림."""
    return [{
        "task_id": r["task_id"],
        "source": r["source"],
        "original_question": r["original_question"],
        "task": r["task"],
        "gold_answer": r["gold_answer"],
        "gold_modality": r["gold_modality"],
        "image_path": r["image_path"],
        "metadata": r["metadata"],
    } for r in items]


def build_records() -> list[dict]:
    comi = json.load(open(COMI_FULL))["results"]

    unchanged = [r for r in comi if r["source"] in UNCHANGED_SOURCES]
    print(f"[D] 기존 유지 5개 소스: {len(unchanged)}건")
    gsm_items = [r for r in unchanged if r["source"] == "gsm8k"]
    recover_gsm8k_gold(gsm_items)
    unchanged_clean = clean_unchanged(unchanged)

    mmlu_new = normalize_mmlu_openstyle(json.load(open(MMLU_OPENSTYLE)))
    print(f"[A] mmlu(open-style): {len(mmlu_new)}건")

    sqa_new = normalize_scienceqa_409(json.load(open(SCIENCEQA_409)))
    print(f"[B] ScienceQA(필터링): {len(sqa_new)}건")

    vqa_new = normalize_vqav2_991(json.load(open(VQAV2_991))["results"])
    print(f"[C] VQAv2(blind-test): {len(vqa_new)}건")

    final = unchanged_clean + mmlu_new + sqa_new + vqa_new
    print(f"\n합계: {len(final)}건 (기대값 5,682)")
    assert len(final) == 5682, f"개수 불일치: {len(final)}"

    task_ids = [r["task_id"] for r in final]
    assert len(task_ids) == len(set(task_ids)), "task_id 중복 발견"

    print("\n소스별 최종 개수:")
    for s, c in Counter(r["source"] for r in final).most_common():
        print(f"  {s:14s} {c}")

    return final


def to_stage1_sample(rec: dict) -> dict:
    """stage1_judge.py가 요구하는 최소 스키마(id/prompt/task/source/note)를 채우되,
    나머지 필드(gold_answer/gold_modality/image_path/metadata)도 그대로 실어 보낸다.
    stage1/stage2 둘 다 `{**sample, ...}`로 원본 필드를 보존하며 지나가므로 안전하다."""
    note = NOTE_BY_SOURCE_GROUP.get(rec["source"], DEFAULT_NOTE)
    return {
        "id": rec["task_id"],
        "prompt": rec["original_question"],
        "task": rec["task"].upper(),  # stage1의 TASK_MODALITY 키가 대문자(T2T/TI2T/T2I/TI2I)
        "source": rec["source"],
        "note": note,
        # 아래는 stage1_judge.py 입장에서는 pass-through 필드
        "gold_answer": rec["gold_answer"],
        "gold_modality": rec["gold_modality"],
        "image_path": rec["image_path"],
        "metadata": rec["metadata"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON,
                     help="rich json 출력 경로(gold_answer/image_path/metadata 포함)")
    ap.add_argument("--out-jsonl", default=DEFAULT_OUT_JSONL,
                     help="stage1_judge.py --input 으로 바로 쓸 수 있는 jsonl 출력 경로")
    args = ap.parse_args()

    records = build_records()

    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"total_questions": len(records), "datasets": sorted(set(r["source"] for r in records)),
         "results": records},
        open(out_json_path, "w"), ensure_ascii=False, indent=2,
    )
    print(f"\n[rich json] 저장: {out_json_path}")

    out_jsonl_path = Path(args.out_jsonl)
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(to_stage1_sample(rec), ensure_ascii=False) + "\n")
    print(f"[stage1 입력 jsonl] 저장: {out_jsonl_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
