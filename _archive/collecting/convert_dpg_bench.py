#!/usr/bin/env python3
"""Convert DPG-Bench individual .txt prompt files into a single prompts.jsonl."""

import json
import os
from pathlib import Path

SRC = Path(__file__).parent / "raw" / "t2i" / "dpg_bench" / "prompts"
OUT = Path(__file__).parent / "raw" / "t2i" / "dpg_bench" / "prompts.jsonl"

# If raw/t2i/dpg_bench is not writable, write next to this script
if not os.access(OUT.parent, os.W_OK):
    OUT = Path(__file__).parent / "dpg_bench_prompts.jsonl"
    print(f"[info] dpg_bench dir not writable, writing to {OUT}")

records = []
for f in sorted(SRC.glob("*.txt")):
    prompt = f.read_text(encoding="utf-8").strip()
    records.append({"id": f.stem, "prompt": prompt})

records.sort(key=lambda r: r["id"])

with open(OUT, "w", encoding="utf-8") as fout:
    for r in records:
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[done] {len(records)} prompts -> {OUT}")
