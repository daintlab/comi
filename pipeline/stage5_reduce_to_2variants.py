"""
Stage 5: variant 5개 -> 2개(의문형 1 + 비의문형 1)로 축소.

2026-09 미팅에서 variant 수를 5개에서 2개(의문형/비의문형 각 1개)로 줄이기로
결정. Stage 3 결과(checks[i].passed, slot_assigned)만 있으면 기계적으로 축소
가능하지만, 다음 세 경우는 자동으로 못 정한다:

  a) 5개 다 fail
  b) interrogative 생존자가 0개
  c) non-interrogative 생존자가 0개

이 케이스들은 별도로 뽑아서(needs_inspection) 사람이 직접 보고
  - rescue: 실패 판정이어도 실제로는 써도 되겠다 싶은 variant를 의문형 1 +
            비의문형 1로 직접 지정해서 구제
  - drop:   샘플 자체를 드롭
을 결정한다 (scripts/variant2_inspection_app.py, 결정 로그는
decisions/variant2_reduction_manual_decisions.json).

실패 사유가 명확한 단일 원인(예: MagicBrush의 "제공된 이미지 편집 -> 새 이미지
생성" 드리프트, dpg_bench의 "원본이 요구한 디테일 누락")으로 전부 설명되는
샘플은 사람이 일일이 안 봐도 자동으로 drop 처리했다
(decisions/variant2_reduction_auto_drop_registry.json). 사람이 직접 보고 다른
결정을 내린 경우 사람 결정이 항상 우선한다.

사용법
------
    python stage5_reduce_to_2variants.py --input ../data/stage3_full_5682.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

SEED = 42


def split_by_slot(data: dict) -> tuple[list[dict], list[dict]]:
    """5-variant 결과를, 기계적으로 2개로 줄일 수 있는 것과 없는 것으로 나눈다."""
    rng = random.Random(SEED)
    resolved, needs_inspection = [], []

    for r in data["results"]:
        checks = r["checks"]
        passed_idx = [i for i, c in enumerate(checks) if c["passed"]]
        interrog_idx = [i for i in passed_idx if checks[i].get("slot_assigned") == "interrogative"]
        noninterrog_idx = [i for i in passed_idx if checks[i].get("slot_assigned") == "non-interrogative"]

        if len(passed_idx) == 0:
            reason = "all_failed"
        elif len(interrog_idx) == 0:
            reason = "no_interrogative_survivor"
        elif len(noninterrog_idx) == 0:
            reason = "no_noninterrogative_survivor"
        else:
            reason = None

        if reason:
            rec = dict(r)
            rec["inspect_reason"] = reason
            rec["n_pass"] = len(passed_idx)
            needs_inspection.append(rec)
            continue

        pick_i, pick_n = rng.choice(interrog_idx), rng.choice(noninterrog_idx)
        rec = dict(r)
        rec["variants"] = [r["variants"][pick_i], r["variants"][pick_n]]
        rec["checks"] = [checks[pick_i], checks[pick_n]]
        rec["n_pass"] = 2
        rec["picked_variant_indices"] = [pick_i, pick_n]  # 원본 5개 중 몇 번째였는지, 감사용
        resolved.append(rec)

    return resolved, needs_inspection


def apply_manual_and_auto_decisions(
    needs_inspection: list[dict], manual_decisions: dict, auto_drop_ids: set[str]
) -> tuple[list[dict], list[str]]:
    """사람 결정 최우선, 그다음 자동 드롭 레지스트리. 아무 데도 없으면 에러."""
    by_id = {r["id"]: r for r in needs_inspection}
    rescued, dropped, undecided = [], [], []

    for tid, rec in by_id.items():
        dec = manual_decisions.get(tid)
        if dec and dec.get("decision") == "rescue":
            i, j = dec["picked_indices"]
            new_rec = dict(rec)
            new_rec["variants"] = [rec["variants"][i], rec["variants"][j]]
            new_rec["checks"] = [rec["checks"][i], rec["checks"][j]]
            new_rec["n_pass"] = sum(1 for k in (i, j) if rec["checks"][k]["passed"])
            new_rec["picked_variant_indices"] = [i, j]
            new_rec["rescued_manually"] = True
            new_rec["rescue_memo"] = dec.get("memo", "")
            new_rec.pop("inspect_reason", None)
            rescued.append(new_rec)
        elif dec and dec.get("decision") == "drop":
            dropped.append(tid)
        elif tid in auto_drop_ids:
            dropped.append(tid)
        else:
            undecided.append(tid)

    assert not undecided, f"결정 안 된 샘플이 남아있음, 확인 필요: {undecided}"
    return rescued, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/stage3_full_5682.json")
    ap.add_argument("--manual-decisions", default="decisions/variant2_reduction_manual_decisions.json")
    ap.add_argument("--auto-drop-registry", default="decisions/variant2_reduction_auto_drop_registry.json")
    ap.add_argument("--output", default=None,
                     help="기본값: ../data/stage3_2variant_final.json")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    resolved, needs_inspection = split_by_slot(data)

    manual_decisions = json.loads(Path(args.manual_decisions).read_text()) if Path(args.manual_decisions).exists() else {}
    auto_drop_ids = set()
    if Path(args.auto_drop_registry).exists():
        auto_drop_ids = {r["id"] for r in json.loads(Path(args.auto_drop_registry).read_text())["results"]}

    rescued, dropped = apply_manual_and_auto_decisions(needs_inspection, manual_decisions, auto_drop_ids)

    final = dict(data)
    final["results"] = resolved + rescued
    final["n_variants"] = 2
    final["slots"] = ["interrogative", "non-interrogative"]
    final["report"] = {
        "n_samples": len(final["results"]),
        "reduction": {
            "original_n_samples": len(data["results"]),
            "auto_resolved": len(resolved),
            "needed_inspection": len(needs_inspection),
            "rescued_manually": len(rescued),
            "dropped": len(dropped),
            "needs_inspection_reasons": dict(Counter(r["inspect_reason"] for r in needs_inspection)),
        },
    }

    out = args.output or "../data/stage3_2variant_final.json"
    Path(out).write_text(json.dumps(final, ensure_ascii=False, indent=2))

    red = final["report"]["reduction"]
    print(f"원본 {red['original_n_samples']}건 (5-variant)")
    print(f"  -> 자동 2-variant 축소: {red['auto_resolved']}건")
    print(f"  -> 수동 검토 필요: {red['needed_inspection']}건 {red['needs_inspection_reasons']}")
    print(f"     - 구제(rescue): {red['rescued_manually']}건")
    print(f"     - 드롭: {red['dropped']}건")
    print(f"최종: {final['report']['n_samples']}건 -> {out}")


if __name__ == "__main__":
    main()
