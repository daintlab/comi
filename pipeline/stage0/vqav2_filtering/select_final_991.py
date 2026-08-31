"""vqav2_blindtest_survived_1541.json(blind test 생존 1,541건)에서 무작위로 991개를
뽑아 최종 TI2T 세트로 확정한다. 나머지 550개는 별도 저장(재현/추후 활용 대비).
SEED 고정, 이 파일 자체가 영구 스크립트.

사용법:
    python select_final_991.py --src vqav2_blindtest_survived_1541.json \\
        --out-selected vqav2_final_ti2t_991.json --out-excluded vqav2_excluded_550.json
"""
import argparse
import json
import random
from collections import Counter

SEED = 42
N_SELECT = 991


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-selected", required=True)
    ap.add_argument("--out-excluded", required=True)
    args = ap.parse_args()
    SRC, OUT_SELECTED, OUT_EXCLUDED = args.src, args.out_selected, args.out_excluded

    d = json.load(open(SRC))
    items = d["results"]
    assert len(items) == 1541, f"기대한 1541건이 아니라 {len(items)}건임"

    items_sorted = sorted(items, key=lambda r: r["task_id"])  # 결정적 순서 고정
    rng = random.Random(SEED)
    selected = rng.sample(items_sorted, N_SELECT)
    selected_ids = {r["task_id"] for r in selected}
    excluded = [r for r in items_sorted if r["task_id"] not in selected_ids]

    assert len(selected) == N_SELECT
    assert len(excluded) == len(items) - N_SELECT
    assert len(selected_ids & {r["task_id"] for r in excluded}) == 0

    json.dump({"total_questions": len(selected), "datasets": ["VQAv2"], "results": selected},
              open(OUT_SELECTED, "w"), ensure_ascii=False, indent=2)
    json.dump({"total_questions": len(excluded), "datasets": ["VQAv2"], "results": excluded},
              open(OUT_EXCLUDED, "w"), ensure_ascii=False, indent=2)

    print(f"선정: {len(selected)}건 -> {OUT_SELECTED}")
    print(f"제외: {len(excluded)}건 -> {OUT_EXCLUDED}")

    # ---- 선정본 분포 ----
    at = Counter(r["metadata"]["answer_type"] for r in selected)
    print("\n선정 991건 answer_type 분포:")
    for t, c in at.most_common():
        print(f"  {t:8s} {c:4d} ({c/len(selected)*100:.1f}%)")

    qt = Counter(r["metadata"]["question_type"] for r in selected)
    print(f"\n선정 991건 question_type 분포 (전체 {len(qt)}종, 상위 15개):")
    for t, c in qt.most_common(15):
        print(f"  {t:20s} {c:4d} ({c/len(selected)*100:.1f}%)")

    return selected, excluded


if __name__ == "__main__":
    main()
