#!/usr/bin/env python3
"""Modality Selection Accuracy (MSA) / Task Performance on correct-modality
responses (TP+) / COMI-aware task score (COMI-score) over COMI-Bench model
outputs -- the three metrics reported in the paper.

Definitions:

  MSA: for a text-gold item, success iff the response's text content is
    non-empty -- regardless of what `pred_modality` says (a response can
    contain an image segment *and* non-empty text; that still counts as a
    text-modality success, since the paper's response-modality classifier
    treats text and special-token signals as independent and MSA only
    requires the gold modality to be *among* the detected modalities). For
    an image-gold item, success iff `pred_modality == "image"`.

  TP+: mean per-item score (0-100), over MSA-passing items only (`I+` in
    the paper).

  COMI-score: mean per-item score (0-100) over *all* items -- MSA-passing
    items keep their real score, MSA-failing items are scored as `s_min`
    (dataset-specific: 50 for MagicBrush's 0-100 CLIP-I scale -- the paper
    empirically sets the CLIP-I floor to 0.50 rather than the metric's
    theoretical -1.0, since unrelated-image pairs average ~0.52 similarity;
    0 for every other source).

Expected input (one results jsonl per model, produced by `inference/` and
then scored per-source -- see root README.md, "Evaluation"): each line is a JSON
object with at least:
    variant_id, source, gold_modality ("text" or "image"),
    pred_modality ("text" or "image"), text_output (string)
plus a per-item `score` field (0-100) added by a source-specific scorer
(see rule_based_scorers.py for the four sources that don't need a judge
model; the other four need a judge -- see root README.md, "Evaluation").

Usage:
    python compute_msa_tp.py --input results_scored.jsonl --out table.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Per-source score for an MSA-failing item (the paper's `s_min`), on the
# 0-100 scale every `score` field in the input is expected to already be on.
S_MIN = defaultdict(lambda: 0.0)
S_MIN["MagicBrush"] = 50.0

# Paper's own source labels: the OSQ-bench column is MMLU converted to
# open-style questions via the OSQ-bench procedure (Myrzakhan et al., 2024)
# -- same underlying dataset as `rule_based_scorers.py`'s "mmlu".
SOURCE_ORDER = ["gsm8k", "mmlu", "ScienceQA", "VQAv2",
                "geneval", "dpg_bench", "MagicBrush", "ImgEditBench"]
SOURCE_LABEL = {"mmlu": "OSQ-bench"}


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

        tp_plus = (sum(it["score"] for it in passed) / len(passed)) if passed else None

        s_min = S_MIN[source]
        comi_scores = [it["score"] if msa_pass(it) else s_min for it in its]
        comi_score = sum(comi_scores) / n if n else None

        rows[source] = {"n": n, "n_msa_pass": len(passed), "MSA": msa,
                         "TP+": tp_plus, "COMI-score": comi_score}
    return rows


def format_markdown(rows: dict[str, dict]) -> str:
    out = ["| Source | n | MSA (%) | TP+ | COMI-score |", "|---|---:|---:|---:|---:|"]
    for source in SOURCE_ORDER:
        if source not in rows:
            continue
        r = rows[source]
        tp_s = f"{r['TP+']:.1f}" if r["TP+"] is not None else "N/A"
        comi_s = f"{r['COMI-score']:.1f}" if r["COMI-score"] is not None else "N/A"
        label = SOURCE_LABEL.get(source, source)
        out.append(f"| {label} | {r['n']} | {r['MSA']:.1f} | {tp_s} | {comi_s} |")
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
            f"run a source-specific scorer first (see root README.md, 'Evaluation'). "
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
