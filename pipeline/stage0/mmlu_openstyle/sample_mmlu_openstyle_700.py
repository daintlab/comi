"""OSQ-bench(Open-Style/Open-LLM-Benchmark)에서 MMLU open-style 문항 700개를,
원본 MMLU 57개 subject의 pool 내 비율을 그대로 유지한 채 층화 샘플링한다.

배경:
- Open-LLM-Leaderboard 논문(arXiv:2406.07545)의 2단계 필터링 결과를 그대로 사용.
  Stage 1(first_filter): 이진 분류로 "선택지 없이도 답 가능"한 문항을 1차 선별.
  Stage 2(second_filter): Stage 1에서 NO로 분류된 문항 중, 1~10점 신뢰도 점수가
  5점 이상이면 "구제"해서 최종 후보에 합류시킴 (논문 3장 Filtering 참고).
  실제 확인 결과 공개 데이터셋의 NO 그룹은 이미 second_filter>=5만 남아있어서
  (5 미만 0건), eligible pool = 전체 MMLU 7,784건과 동일함.
- 원본 comi_bench의 mmlu 표본은 subject 메타데이터가 없어서, cais/mmlu(공식
  MMLU)와 문항 텍스트를 대조해 subject를 100% 복원한 뒤 그 비율대로 700개를
  배분(largest-remainder 방식)했다.

출력: mmlu_openstyle_700.json (700건, subject 필드 포함)
"""
import json
import random
from collections import defaultdict

import datasets

N_TARGET = 700
SEED = 42
OUT_PATH = "mmlu_openstyle_700.json"


def main():
    osq = datasets.load_dataset("Open-Style/Open-LLM-Benchmark", "questions")["train"]
    osq_mmlu = osq.filter(
        lambda x: x["dataset"] == "MMLU" and (x["first_filter"] == "YES" or x["second_filter"] >= 5)
    )
    print(f"OSQ eligible mmlu pool: {len(osq_mmlu)}")

    mmlu_ref = datasets.load_dataset("cais/mmlu", "all")
    q2subj = {}
    for split in ["test", "validation", "dev"]:
        for row in mmlu_ref[split]:
            q2subj[row["question"]] = row["subject"]

    by_subject = defaultdict(list)
    unmatched = 0
    for row in osq_mmlu:
        subj = q2subj.get(row["question"])
        if subj is None:
            unmatched += 1
            continue
        by_subject[subj].append(row)
    print(f"subject 매칭 실패: {unmatched}건 (0이어야 정상)")

    total = sum(len(v) for v in by_subject.values())
    raw = {s: N_TARGET * len(v) / total for s, v in by_subject.items()}
    base = {s: int(raw[s]) for s in raw}
    remainder = N_TARGET - sum(base.values())
    frac_order = sorted(raw.keys(), key=lambda s: -(raw[s] - base[s]))
    for s in frac_order[:remainder]:
        base[s] += 1
    assert sum(base.values()) == N_TARGET

    random.seed(SEED)
    out = []
    for s, n in base.items():
        pool = by_subject[s][:]
        random.shuffle(pool)
        for row in pool[:n]:
            gold_text = next(o["text"] for o in row["options"] if o["label"] == row["answerKey"])
            out.append({
                "source": "mmlu",
                "task": "t2t",
                "original_question": row["question"],
                "gold_answer": gold_text,
                "gold_answer_key": row["answerKey"],
                "mmlu_subject": s,
                "first_filter": row["first_filter"],
                "second_filter": row["second_filter"],
            })

    random.shuffle(out)
    for i, r in enumerate(out):
        r["task_id"] = f"mmlu_openstyle_{i:05d}"

    json.dump(out, open(OUT_PATH, "w"), ensure_ascii=False, indent=2)
    print(f"saved {len(out)} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
