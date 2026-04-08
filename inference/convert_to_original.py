#!/usr/bin/env python3
"""Convert v3.json using ORIGINAL prompts (no refinement) for Exp 4 comparison."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def convert_item(r: Dict[str, Any]) -> Dict[str, Any]:
    input_segs: List[Dict[str, str]] = []
    if r.get("image_path"):
        input_segs.append({"type": "image", "content": r["image_path"]})
    input_segs.append({"type": "text", "content": r["original_question"]})

    return {
        "task_id": f"{r['task_id']}_orig",
        "source": r["source"],
        "benchmark": r.get("source", ""),
        "expectation": r.get("gold_modality", "text"),
        "task": r.get("task", ""),
        "original_task_id": r["task_id"],
        "variant_idx": -1,
        "input": input_segs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = data.get("results", data)
    items = [convert_item(r) for r in results]

    Path(args.output).write_text(
        json.dumps({"items": items}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {len(items)} items written to {args.output}")


if __name__ == "__main__":
    main()
