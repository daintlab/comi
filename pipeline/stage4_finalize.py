"""
Stage 4: 큐레이션 확정 (drop 결정 반영, 최종 데이터셋 생성).

Stage 3 산출물(checks/n_pass)만으로는 "얼마나 실패했는지"는 알아도 "그래서
드롭할지 말지"는 알 수 없다. 이 단계는 사람이 내린 드롭 결정
(decisions/drop_decisions.json, decisions/qa_decisions.json)을 실제로 반영해서
평가에 쓸 최종 데이터셋을 만든다.

결정 파일 스키마 (두 파일 공통)
    {id: {"decision": "drop" | "keep", "drop_variants": [1-indexed, ...]}}

규칙
----
  decision == "drop"                        -> 샘플 전체 제거
  decision == "keep" and drop_variants 있음  -> 해당 variant만 variants/checks에서
                                                제거. 전부 제거되면(남는 게 0개)
                                                샘플 자체를 제거로 취급
  decision == "keep" and drop_variants 없음  -> 그대로 유지 (검토했지만 문제 없다고
                                                판단한 것)
  결정 파일에 없는 나머지                     -> 그대로 유지

드롭 사유는 크게 두 갈래였다:
  - 소스 자체 손상/라벨 불일치 (예: 소스 텍스트 단어 중복, task 라벨과 실제
    요구사항 불일치) -> 샘플 전체 드롭
  - 룰베이스(exact-match) 채점 소스(gsm8k 등)에서 variant가 원본 문제의 수량/
    관계를 바꿔 정답 자체가 달라지는 경우 -> 문제 되는 variant만 드롭. 이미지
    편집처럼 모델/판정기가 실제 출력을 보고 채점하는 소스는 프롬프트가 조금
    흔들려도 자기교정이 되지만, 고정된 gold_answer를 그대로 재사용하는 소스는
    그렇지 않기 때문.

사용법
------
    python stage4_finalize.py --input ../data/stage3_full_5682.json
    python stage4_finalize.py --input ... --decisions decisions/drop_decisions.json decisions/qa_decisions.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_merged_decisions(decision_paths: list[str]) -> dict:
    merged: dict = {}
    for path in decision_paths:
        decisions = json.loads(Path(path).read_text())
        overlap = set(decisions) & set(merged)
        assert not overlap, f"{path}: 다른 결정 파일과 id가 겹침, 확인 필요: {overlap}"
        for tid, dec in decisions.items():
            merged[tid] = {
                "decision": dec.get("decision"),
                "drop_variants": dec.get("drop_variants", []),
            }
    return merged


def finalize(data: dict, decisions: dict) -> dict:
    original_n = len(data["results"])

    full_drops: list[str] = []
    partial_drops: dict[str, list[int]] = {}
    emptied_by_partial: list[str] = []
    reviewed_kept_unchanged: list[str] = []

    final_results = []
    for r in data["results"]:
        tid = r["id"]
        dec = decisions.get(tid)

        if dec is None:
            final_results.append(r)
            continue

        if dec["decision"] == "drop":
            full_drops.append(tid)
            continue

        if dec["decision"] == "keep" and dec.get("drop_variants"):
            drop_idx = set(dec["drop_variants"])  # 1-indexed
            keep_pos = [i for i in range(len(r["checks"])) if (i + 1) not in drop_idx]

            if not keep_pos:
                emptied_by_partial.append(tid)
                continue

            r = dict(r)
            r["checks"] = [r["checks"][i] for i in keep_pos]
            if r.get("variants"):
                r["variants"] = [r["variants"][i] for i in keep_pos]
            r["n_pass"] = sum(1 for c in r["checks"] if c["passed"])
            r["curation_note"] = f"variant {sorted(drop_idx)} dropped"

            partial_drops[tid] = sorted(drop_idx)
            final_results.append(r)
            continue

        reviewed_kept_unchanged.append(tid)
        final_results.append(r)

    data["results"] = final_results
    data["report"]["n_samples"] = len(final_results)
    data["report"]["curation"] = {
        "original_n_samples": original_n,
        "final_n_samples": len(final_results),
        "full_drops": sorted(full_drops + emptied_by_partial),
        "partial_drops": dict(sorted(partial_drops.items())),
        "reviewed_kept_unchanged": sorted(reviewed_kept_unchanged),
    }
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/stage3_full_5682.json")
    ap.add_argument("--decisions", nargs="+",
                     default=["decisions/drop_decisions.json", "decisions/qa_decisions.json"])
    ap.add_argument("--output", default=None,
                     help="기본값: ../data/stage3_final_<N>.json (N=최종 샘플 수)")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    decisions = load_merged_decisions(args.decisions)
    data = finalize(data, decisions)

    n = data["report"]["n_samples"]
    out = args.output or f"../data/stage3_final_{n}.json"
    Path(out).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    c = data["report"]["curation"]
    print(f"원본 {c['original_n_samples']}건 -> 최종 {c['final_n_samples']}건")
    print(f"전체 드롭: {len(c['full_drops'])}건")
    print(f"부분 드롭(variant만): {len(c['partial_drops'])}건")
    print(f"리뷰 후 유지: {len(c['reviewed_kept_unchanged'])}건")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
