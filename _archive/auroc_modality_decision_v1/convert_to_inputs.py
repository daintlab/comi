#!/usr/bin/env python3
"""
Convert v3.json (aug_bench_v2 pipeline format) to the input format
expected by run_inputs_single_load.py.

v3.json format:
  { "results": [ { task_id, source, task, original_question, gold_modality,
                    image_path, refined_variants_kept, ... } ] }

Output format:
  { "items": [ { task_id, source, benchmark, expectation,
                  input: [ {type, content}, ... ] } ] }

For each item, we use the first refined_variants_kept (or original_question
if no variants exist) as the prompt text.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def convert_item(r: Dict[str, Any], variant_idx: int = 0, prompt: str = "") -> Dict[str, Any]:
    """Convert a single v3 result + variant to the run_inputs_single_load format."""
    input_segs: List[Dict[str, str]] = []

    if r.get("image_path"):
        input_segs.append({"type": "image", "content": r["image_path"]})

    input_segs.append({"type": "text", "content": prompt})

    return {
        "task_id": f"{r['task_id']}_v{variant_idx}",
        "source": r["source"],
        "benchmark": r.get("source", ""),
        "expectation": r.get("gold_modality", "text"),
        "task": r.get("task", ""),
        "original_task_id": r["task_id"],
        "variant_idx": variant_idx,
        "input": input_segs,
    }


def expand_item(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a v3 result into one item per kept variant."""
    variants = r.get("refined_variants_kept", [])
    if not variants:
        # Fallback to original question
        return [convert_item(r, variant_idx=0, prompt=r["original_question"])]

    return [
        convert_item(r, variant_idx=i, prompt=v)
        for i, v in enumerate(variants)
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to v3.json")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Filter by task type (t2t, ti2t, t2i, ti2i). Default: all.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items per task type.",
    )
    args = p.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = data.get("results", data)

    # Filter by task type
    if args.tasks:
        results = [r for r in results if r.get("task") in args.tasks]

    # Limit per task
    if args.limit is not None:
        from collections import defaultdict
        counts = defaultdict(int)
        filtered = []
        for r in results:
            t = r.get("task", "")
            if counts[t] < args.limit:
                filtered.append(r)
                counts[t] += 1
        results = filtered

    items = []
    for r in results:
        items.extend(expand_item(r))

    output = {"items": items}
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {len(items)} items written to {args.output}")


if __name__ == "__main__":
    main()
