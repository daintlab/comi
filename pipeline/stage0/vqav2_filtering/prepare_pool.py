"""VQAv2 val split에서 answer_type(yes/no, number, other) 층화 샘플링으로 blind test
입력 pool을 만든다. yes/no 비중은 전체의 1/3로 캡(원본 자연 비율은 37.6%로 그것보다 살짝
낮춰서, blind test 탈락률이 yes/no 쪽으로 과도하게 쏠리는 걸 방지).

number/other는 yes/no를 뺀 나머지 2/3 안에서, 원본 자연 비율(number:other ≈ 21:79)을
그대로 유지해서 배분한다(임의로 50:50 안 하고, 실제 VQAv2 분포를 최대한 보존).

largest-remainder method로 정확히 N개를 맞춘다. SEED 고정, 재현 가능.

사용법:
    python prepare_pool.py --n 200 --out vqav2_audit_pilot200.json --seed 42
    python prepare_pool.py --n 3000 --out vqav2_audit_full.json --seed 42 --exclude-qids <prev pilot json>
"""
import argparse
import json
import random
import os
from collections import defaultdict

# 공식 VQAv2 val2014 questions/annotations -- https://visualqa.org/download.html 에서
# 받아서 로컬에 두고 경로를 env var로 지정 (기본값은 이 스크립트 옆 vqav2_official/ 하위).
_DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "vqav2_official")
QUESTIONS_PATH = os.getenv(
    "VQAV2_QUESTIONS_PATH", os.path.join(_DEFAULT_DIR, "v2_OpenEnded_mscoco_val2014_questions.json"))
ANNOTATIONS_PATH = os.getenv(
    "VQAV2_ANNOTATIONS_PATH", os.path.join(_DEFAULT_DIR, "v2_mscoco_val2014_annotations.json"))

YESNO_CAP_FRACTION = 1.0 / 3.0


def largest_remainder(target_fracs: dict, n: int) -> dict:
    """{key: fraction} -> {key: 정수 개수}, 합이 정확히 n이 되도록 largest-remainder 배분."""
    raw = {k: f * n for k, f in target_fracs.items()}
    floors = {k: int(v) for k, v in raw.items()}
    remainder = n - sum(floors.values())
    order = sorted(raw.keys(), key=lambda k: raw[k] - floors[k], reverse=True)
    for k in order[:remainder]:
        floors[k] += 1
    return floors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["stratified", "random"], default="stratified",
                     help="stratified: answer_type별 층화(yes/no 1/3 캡). "
                          "random: 층화 없이 val 전체에서 단순 무작위.")
    ap.add_argument("--exclude-question-ids", nargs="*", default=[],
                     help="이미 파일럿 등에서 쓴 question_id 리스트(json 파일 경로들) -- 중복 방지용")
    args = ap.parse_args()

    questions = json.load(open(QUESTIONS_PATH))["questions"]
    annotations = json.load(open(ANNOTATIONS_PATH))["annotations"]
    by_qid_ann = {a["question_id"]: a for a in annotations}
    by_qid_q = {q["question_id"]: q for q in questions}

    exclude = set()
    for path in args.exclude_question_ids:
        prev = json.load(open(path))
        items = prev if isinstance(prev, list) else prev.get("results", prev)
        for it in items:
            qid = it.get("question_id")
            if qid is not None:
                exclude.add(qid)
    if exclude:
        print(f"제외할 기존 question_id: {len(exclude)}개")

    by_type = defaultdict(list)
    for q in questions:
        qid = q["question_id"]
        if qid in exclude:
            continue
        ans_type = by_qid_ann[qid]["answer_type"]
        by_type[ans_type].append(qid)

    total_pool = sum(len(v) for v in by_type.values())
    print(f"전체 pool(중복 제외 후): {total_pool}건")
    for t, ids in by_type.items():
        print(f"  {t:8s} {len(ids):8d} ({len(ids)/total_pool*100:.1f}%)")

    rng = random.Random(args.seed)

    if args.mode == "random":
        # 층화 없이 전체 pool에서 단순 무작위 -- 자연 분포 그대로 반영됨
        all_qids = sorted(q for ids in by_type.values() for q in ids)
        sampled_qids = rng.sample(all_qids, args.n)
        print(f"\n[random 모드] 층화 없이 단순 무작위 {args.n}건 샘플링")
    else:
        # 목표 비율: yes/no = 1/3 고정, 나머지 2/3는 number:other를 원본 자연비율대로 배분
        non_yn_natural = {t: len(ids) for t, ids in by_type.items() if t != "yes/no"}
        non_yn_total = sum(non_yn_natural.values())
        target_fracs = {"yes/no": YESNO_CAP_FRACTION}
        for t, cnt in non_yn_natural.items():
            target_fracs[t] = (1 - YESNO_CAP_FRACTION) * (cnt / non_yn_total)

        print("\n목표 비율:", {k: round(v, 4) for k, v in target_fracs.items()})

        target_counts = largest_remainder(target_fracs, args.n)
        print("목표 개수:", target_counts, "합계:", sum(target_counts.values()))

        sampled_qids = []
        for t, k in target_counts.items():
            pool = sorted(by_type[t])  # 결정적 순서 고정
            if k > len(pool):
                raise ValueError(f"{t} pool({len(pool)})이 목표({k})보다 작음")
            sampled_qids.extend(rng.sample(pool, k))
    rng.shuffle(sampled_qids)

    items = []
    for qid in sampled_qids:
        q = by_qid_q[qid]
        a = by_qid_ann[qid]
        items.append({
            "question_id": qid,
            "image_id": q["image_id"],
            "question": q["question"],
            "answers": a["answers"],  # 10명 어노테이터 원본 그대로(answer/answer_confidence/answer_id)
            "multiple_choice_answer": a["multiple_choice_answer"],
            "answer_type": a["answer_type"],
            "question_type": a["question_type"],
        })

    out_path = args.out
    json.dump(items, open(out_path, "w"), ensure_ascii=False, indent=2)

    actual_counts = defaultdict(int)
    for it in items:
        actual_counts[it["answer_type"]] += 1
    print(f"\n실제 샘플링 결과 (n={len(items)}):")
    for t, c in actual_counts.items():
        print(f"  {t:8s} {c:5d} ({c/len(items)*100:.1f}%)")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
