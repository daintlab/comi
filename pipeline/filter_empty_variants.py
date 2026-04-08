#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from copy import deepcopy
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument(
        "--drop_missing_key",
        action="store_true",
        help="If set, also drop items missing refined_variants_kept key.",
    )
    args = ap.parse_args()

    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "results" not in data or not isinstance(data["results"], list):
        raise ValueError("input_json must be a dict with results[]")

    out = deepcopy(data)
    kept_results = []
    dropped_empty = 0
    dropped_missing = 0

    for r in out["results"]:
        if not isinstance(r, dict):
            continue

        if "refined_variants_kept" not in r:
            if args.drop_missing_key:
                dropped_missing += 1
                continue
            kept_results.append(r)
            continue

        v = r.get("refined_variants_kept")
        if isinstance(v, list) and len(v) == 0:
            dropped_empty += 1
            continue

        kept_results.append(r)

    out["results"] = kept_results
    out["total_questions"] = len(kept_results)

    # Refresh datasets list to match remaining sources (stable order by first appearance)
    seen = set()
    datasets = []
    for r in kept_results:
        src = r.get("source")
        if isinstance(src, str) and src not in seen:
            seen.add(src)
            datasets.append(src)
    out["datasets"] = datasets

    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "input_total": len(data["results"]),
                "dropped_empty_refined_variants_kept": dropped_empty,
                "dropped_missing_refined_variants_kept": dropped_missing,
                "output_total": len(kept_results),
                "output_json": args.output_json,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

