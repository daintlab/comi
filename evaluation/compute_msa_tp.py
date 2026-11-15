#!/usr/bin/env python3
"""Modality Selection Accuracy (MSA, strict) / Task Performance (TP) /
Conditional Task Performance (C-TP) over COMI-Bench model outputs.

These are the metrics reported in the paper. Definitions:

  MSA (strict): for a text-gold item, success iff the response's text
    content is non-empty -- regardless of what `pred_modality` says (a
    response can contain an image segment *and* non-empty text; that still
    counts as a text-modality success). For an image-gold item, success
    iff `pred_modality == "image"`.

  TP: mean per-item score (0-100), over MSA-passing items only.

  C-TP: mean per-item score (0-100) over *all* items -- MSA-passing items
    keep their real score, MSA-failing items are scored as `s_fail`
    (dataset-specific: 50 for MagicBrush's 0-100 CLIP-I scale, 0 for every
    other source).

Expected input (one results jsonl per model, produced by `inference/` and
then scored per-source -- see evaluation/README.md): each line is a JSON
object with at least:
    variant_id, source, gold_modality ("text" or "image"),
    pred_modality ("text" or "image"), text_output (string)
plus a per-item `score` field (0-100) added by a source-specific scorer
(see rule_based_scorers.py for the four sources that don't need a judge
model; the other four need a judge -- see evaluation/README.md).

Usage:
    python compute_msa_tp.py --input results_scored.jsonl --out table.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Per-source score for an MSA-failing item, on the 0-100 scale every
# `score` field in the input is expected to already be on.
S_FAIL = defaultdict(lambda: 0.0)
S_FAIL["MagicBrush"] = 50.0

SOURCE_ORDER = ["gsm8k", "mmlu", "ScienceQA", "VQAv2",
                "geneval", "dpg_bench", "MagicBrush", "ImgEditBench"]


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def msa_pass(item: dict) -> bool:
    if item["gold_modality"] == "image":
        return item["pred_modality"] == "image"
    return bool((item.get("text_output") or "").strip())


def compute_table(items: list[dict]) -> dict[str, dict]:
    by_source = defaultdict(list)
    for it in items:
        by_source[it["source"]].append(it)

    rows = {}
    for source, its in by_source.items():
        n = len(its)
        passed = [it for it in its if msa_pass(it)]
        msa = len(passed) / n * 100 if n else 0.0

        tp = (sum(it["score"] for it in passed) / len(passed)) if passed else None

        s_fail = S_FAIL[source]
        ctp_scores = [it["score"] if msa_pass(it) else s_fail for it in its]
        ctp = sum(ctp_scores) / n if n else None

        rows[source] = {"n": n, "n_msa_pass": len(passed), "MSA": msa,
                         "TP": tp, "C-TP": ctp}
    return rows


def format_markdown(rows: dict[str, dict]) -> str:
    out = ["| Source | n | MSA (%) | TP | C-TP |", "|---|---:|---:|---:|---:|"]
    for source in SOURCE_ORDER:
        if source not in rows:
            continue
        r = rows[source]
        tp_s = f"{r['TP']:.1f}" if r["TP"] is not None else "N/A"
        ctp_s = f"{r['C-TP']:.1f}" if r["C-TP"] is not None else "N/A"
        out.append(f"| {source} | {r['n']} | {r['MSA']:.1f} | {tp_s} | {ctp_s} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                     help="scored results jsonl (see module docstring)")
    ap.add_argument("--out", default=None, help="write the markdown table here too")
    args = ap.parse_args()

    items = load_jsonl(args.input)
    missing_score = [it for it in items if "score" not in it]
    if missing_score:
        raise SystemExit(
            f"{len(missing_score)}/{len(items)} items have no `score` field -- "
            f"run a source-specific scorer first (see evaluation/README.md). "
            f"First missing item: {missing_score[0].get('variant_id')}")

    rows = compute_table(items)
    table = format_markdown(rows)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n")
        print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
