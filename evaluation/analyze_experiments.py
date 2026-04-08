#!/usr/bin/env python3
"""
Unified analysis script for CAMO-Bench experiments.

Usage:
    python analyze_experiments.py \
        --v3 v3.json \
        --models tar:Tar/v3_all_output,anole:anole/v3_all_output,show-o2:show-o2/v3_all_output \
        --models_orig tar:Tar/v3_orig_output,anole:anole/v3_orig_output,show-o2:show-o2/v3_orig_output \
        --out results/

Each model directory should contain item_XXXX/ with meta.json and output_r00.json.

Experiments:
    Exp 1: Modality Decision Accuracy
    Exp 2: Prompt Variant Sensitivity
    Exp 3: Sensitivity vs Output Quality
    Exp 4: Surface-Form Cue Analysis (original vs CAMO prompts)
"""

import argparse
import json
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:
    print("[warn] scikit-learn not installed. AUROC will be skipped.")
    roc_auc_score = None


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_v3(path: str) -> Dict[str, Dict]:
    """Load v3.json and index by task_id."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["task_id"]: r for r in data["results"]}


def load_model_results(result_dir: str, v3_lookup: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """Load all item results from a model output directory."""
    items = []
    for d in sorted(glob.glob(os.path.join(result_dir, "item_*"))):
        meta_f = os.path.join(d, "meta.json")
        out_f = os.path.join(d, "output_r00.json")
        if not os.path.exists(meta_f) or not os.path.exists(out_f):
            continue
        meta = json.load(open(meta_f, encoding="utf-8"))
        output = json.load(open(out_f, encoding="utf-8"))

        # Extract modality prob and actual modality
        modality_prob = None
        for s in output:
            if s.get("type") == "meta":
                # Support both Tar (im_start_prob) and Anole (boi_prob)
                modality_prob = s.get("im_start_prob") or s.get("boi_prob")

        # Determine which modalities are present in output
        has_text = any(
            s.get("type") == "text" and s.get("content", "").strip()
            for s in output
        )
        has_image = any(s.get("type") == "image" for s in output)
        actual_modalities = set()
        if has_text:
            actual_modalities.add("text")
        if has_image:
            actual_modalities.add("image")

        # Parse task_id to get original_task_id and variant_idx
        task_id = meta.get("task_id", "")
        original_task_id = meta.get("original_task_id", task_id)
        variant_idx = meta.get("variant_idx", 0)

        # Fallback: parse from task_id format "xxx_v0", "xxx_orig"
        if "original_task_id" not in meta:
            if task_id.endswith("_orig"):
                original_task_id = task_id[:-5]
                variant_idx = -1
            elif "_v" in task_id:
                parts = task_id.rsplit("_v", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    original_task_id = parts[0]
                    variant_idx = int(parts[1])

        # Resolve expectation from v3_lookup if meta has non-standard values
        expectation = meta.get("expectation", "")
        if expectation not in ("text", "image") and v3_lookup:
            v3 = v3_lookup.get(original_task_id, {})
            expectation = v3.get("gold_modality", expectation)

        items.append({
            "task_id": task_id,
            "original_task_id": original_task_id,
            "variant_idx": variant_idx,
            "source": meta.get("source", ""),
            "expectation": expectation,
            "actual_modalities": actual_modalities,
            "modality_prob": modality_prob,
            "output": output,
            "dir": d,
        })
    return items


def generate_mock_results(v3_lookup: Dict, mode: str = "show-o2") -> List[Dict[str, Any]]:
    """
    Generate mock results for models that use external modality control.
    Show-o2: t2t/ti2t → always text (correct), t2i/ti2i → always text (wrong).
    This simulates a model that cannot generate images without explicit modality flag.
    """
    items = []
    for task_id, v3 in v3_lookup.items():
        task_type = v3.get("task", "")
        gold_modality = v3.get("gold_modality", "text")

        # Show-o2 always outputs text when no modality is specified
        if task_type in ("t2t", "ti2t"):
            actual = "text"
            modality_prob = 0.0
        else:  # t2i, ti2i
            actual = "text"
            modality_prob = 0.0

        # Generate for each variant (0-4)
        variants = v3.get("refined_variants_kept", [])
        if not variants:
            variants = [v3.get("original_question", "")]
        for vi, _ in enumerate(variants):
            items.append({
                "task_id": f"{task_id}_v{vi}",
                "original_task_id": task_id,
                "variant_idx": vi,
                "source": v3.get("source", ""),
                "expectation": gold_modality,
                "actual_modalities": {"text"},
                "modality_prob": modality_prob,
                "output": [
                    {"type": "meta", "im_start_prob": modality_prob},
                    {"type": "text", "content": "(mock: external modality control model)"},
                ],
                "dir": "",
            })
    return items


def parse_models_arg(arg: str) -> Dict[str, str]:
    """Parse 'name:path,name:path,...' into dict."""
    models = {}
    for pair in arg.split(","):
        name, path = pair.split(":", 1)
        models[name.strip()] = path.strip()
    return models


# ──────────────────────────────────────────────
# Exp 1: Modality Decision Accuracy
# ──────────────────────────────────────────────

def _compute_auroc(probs, labels):
    """Compute AUROC, FPR, TPR at best threshold. Returns dict or None."""
    if roc_auc_score is None or len(set(labels)) < 2:
        return None
    auroc = roc_auc_score(labels, probs)
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, probs)
    j = tpr_arr - fpr_arr
    best_idx = np.argmax(j)
    return {
        "auroc": auroc,
        "fpr": float(fpr_arr[best_idx]),
        "tpr": float(tpr_arr[best_idx]),
        "threshold": float(thresholds[best_idx]),
    }


def exp1_modality_accuracy(
    model_results: Dict[str, List[Dict]],
    v3_lookup: Dict[str, Dict],
) -> Dict:
    """Compute AUROC-based modality decision metrics per model."""
    results = {}

    for model_name, items in model_results.items():
        # Use only variant 0 for fair comparison
        v0_items = [it for it in items if it["variant_idx"] == 0]
        if not v0_items:
            v0_items = items

        # Collect probs/labels per input-modality group
        all_probs, all_labels = [], []
        text_input_probs, text_input_labels = [], []   # t2t + t2i (text-only input)
        image_input_probs, image_input_labels = [], []  # ti2t + ti2i (text+image input)

        correct = 0
        total = 0
        by_task = defaultdict(lambda: {"correct": 0, "total": 0})

        for it in v0_items:
            expect = it["expectation"]
            actual_mods = it["actual_modalities"]
            hit = expect in actual_mods
            v3 = v3_lookup.get(it["original_task_id"], {})
            task_type = v3.get("task", "?")

            total += 1
            by_task[task_type]["total"] += 1
            if hit:
                correct += 1
                by_task[task_type]["correct"] += 1

            if it["modality_prob"] is not None:
                label = 1 if expect == "image" else 0
                prob = it["modality_prob"]
                all_probs.append(prob)
                all_labels.append(label)

                if task_type in ("t2t", "t2i"):
                    text_input_probs.append(prob)
                    text_input_labels.append(label)
                elif task_type in ("ti2t", "ti2i"):
                    image_input_probs.append(prob)
                    image_input_labels.append(label)

        # AUROC metrics
        roc_all = _compute_auroc(all_probs, all_labels)
        roc_text_input = _compute_auroc(text_input_probs, text_input_labels)
        roc_image_input = _compute_auroc(image_input_probs, image_input_labels)

        # Per-task accuracy (supplementary)
        task_acc = {}
        for t in ["t2t", "ti2t", "t2i", "ti2i"]:
            d = by_task.get(t, {"correct": 0, "total": 0})
            task_acc[t] = d["correct"] / d["total"] if d["total"] > 0 else None

        results[model_name] = {
            "all": roc_all,
            "text_input": roc_text_input,
            "image_input": roc_image_input,
            "overall_acc": correct / total if total > 0 else 0,
            "total": total,
            "task_acc": task_acc,
        }

    return results


# ──────────────────────────────────────────────
# Exp 2: Prompt Variant Sensitivity
# ──────────────────────────────────────────────

def exp2_variant_sensitivity(
    model_results: Dict[str, List[Dict]],
    v3_lookup: Dict[str, Dict],
) -> Dict:
    """Measure modality decision consistency across prompt variants."""
    results = {}

    for model_name, items in model_results.items():
        # Group by original_task_id
        by_question = defaultdict(list)
        for it in items:
            if it["variant_idx"] >= 0:  # Exclude originals
                by_question[it["original_task_id"]].append(it)

        if not by_question:
            results[model_name] = {"skip": "no variant data"}
            continue

        consistent_count = 0
        flip_count = 0
        entropies = []
        total_questions = 0

        by_task_consistency = defaultdict(lambda: {"consistent": 0, "total": 0})

        for orig_id, variants in by_question.items():
            if len(variants) < 2:
                continue

            total_questions += 1
            modalities = ["image" if "image" in v["actual_modalities"] else "text" for v in variants]
            unique = set(modalities)
            v3 = v3_lookup.get(orig_id, {})
            task_type = v3.get("task", "?")

            by_task_consistency[task_type]["total"] += 1

            if len(unique) == 1:
                consistent_count += 1
                by_task_consistency[task_type]["consistent"] += 1
            else:
                flip_count += 1

            # Entropy
            n = len(modalities)
            img_count = sum(1 for m in modalities if m == "image")
            txt_count = n - img_count
            p_img = img_count / n
            p_txt = txt_count / n
            ent = 0
            if p_img > 0:
                ent -= p_img * np.log2(p_img)
            if p_txt > 0:
                ent -= p_txt * np.log2(p_txt)
            entropies.append(ent)

        task_cons = {}
        for t in ["t2t", "ti2t", "t2i", "ti2i"]:
            d = by_task_consistency.get(t, {"consistent": 0, "total": 0})
            task_cons[t] = d["consistent"] / d["total"] if d["total"] > 0 else None

        # Variant accuracy variance: for each question, compute acc across variants
        variant_acc_vars = []
        for orig_id, variants in by_question.items():
            accs = [1.0 if v["expectation"] in v["actual_modalities"] else 0.0 for v in variants]
            if len(accs) >= 2:
                variant_acc_vars.append(np.var(accs))

        results[model_name] = {
            "total_questions": total_questions,
            "consistency_rate": consistent_count / total_questions if total_questions > 0 else 0,
            "flip_rate": flip_count / total_questions if total_questions > 0 else 0,
            "mean_entropy": float(np.mean(entropies)) if entropies else 0,
            "task_consistency": task_cons,
            "mean_variant_acc_var": float(np.mean(variant_acc_vars)) if variant_acc_vars else 0,
        }

    return results


# ──────────────────────────────────────────────
# Exp 3: Output Quality — Original vs CAMO
# ──────────────────────────────────────────────

def _compute_quality(items: List[Dict], v3_lookup: Dict) -> Dict[str, Dict]:
    """
    Compute per-task quality metrics for a set of items.
    - t2t, ti2t: gold_answer containment accuracy
    - t2i, ti2i: image generation rate (proxy for quality)
    Returns {task_type: {acc, n, ...}}
    """
    by_task = defaultdict(lambda: {"correct": 0, "total": 0})

    for it in items:
        v3 = v3_lookup.get(it["original_task_id"], {})
        task_type = v3.get("task", "?")
        gold = v3.get("gold_answer", "")

        by_task[task_type]["total"] += 1

        if task_type in ("t2t", "ti2t"):
            # Text quality: gold answer containment
            if "text" not in it["actual_modalities"]:
                continue  # wrong modality, score 0
            gen_text = ""
            for s in it["output"]:
                if s.get("type") == "text":
                    gen_text += s.get("content", "")
            if gold and gold.lower().strip() in gen_text.lower():
                by_task[task_type]["correct"] += 1

        elif task_type in ("t2i", "ti2i"):
            # Image quality proxy: did the model actually produce an image?
            if "image" in it["actual_modalities"]:
                by_task[task_type]["correct"] += 1

    result = {}
    for t in ["t2t", "ti2t", "t2i", "ti2i"]:
        d = by_task.get(t, {"correct": 0, "total": 0})
        result[t] = {
            "acc": d["correct"] / d["total"] if d["total"] > 0 else None,
            "correct": d["correct"],
            "n": d["total"],
        }
    return result


def exp3_quality_comparison(
    model_results_camo: Dict[str, List[Dict]],
    model_results_orig: Dict[str, List[Dict]],
    v3_lookup: Dict[str, Dict],
) -> Dict:
    """
    Compare output quality between original benchmark prompts and CAMO prompts.
    - t2t, ti2t: gold_answer containment accuracy
    - t2i, ti2i: image generation rate as proxy
    """
    results = {}

    for model_name in model_results_camo:
        if model_name not in model_results_orig:
            results[model_name] = {"skip": "no original prompt results"}
            continue

        camo_items = model_results_camo[model_name]
        orig_items = model_results_orig[model_name]

        # Use variant 0 for CAMO
        camo_v0 = [it for it in camo_items if it["variant_idx"] == 0]
        if not camo_v0:
            camo_v0 = camo_items

        camo_quality = _compute_quality(camo_v0, v3_lookup)
        orig_quality = _compute_quality(orig_items, v3_lookup)

        by_task = {}
        for t in ["t2t", "ti2t", "t2i", "ti2i"]:
            cq = camo_quality[t]
            oq = orig_quality[t]
            delta = None
            if cq["acc"] is not None and oq["acc"] is not None:
                delta = cq["acc"] - oq["acc"]
            by_task[t] = {
                "orig_acc": oq["acc"],
                "camo_acc": cq["acc"],
                "delta": delta,
                "orig_n": oq["n"],
                "camo_n": cq["n"],
                "metric": "gold containment" if t in ("t2t", "ti2t") else "image gen rate",
            }

        results[model_name] = {"by_task": by_task}

    return results


# ──────────────────────────────────────────────
# Exp 4: Surface-Form Cue Analysis
# ──────────────────────────────────────────────

def exp4_cue_analysis(
    model_results_camo: Dict[str, List[Dict]],
    model_results_orig: Dict[str, List[Dict]],
    v3_lookup: Dict[str, Dict],
) -> Dict:
    """Compare modality accuracy between original and CAMO prompts."""
    results = {}

    for model_name in model_results_camo:
        if model_name not in model_results_orig:
            results[model_name] = {"skip": "no original prompt results"}
            continue

        camo_items = model_results_camo[model_name]
        orig_items = model_results_orig[model_name]

        # Use variant 0 for CAMO
        camo_v0 = {it["original_task_id"]: it for it in camo_items if it["variant_idx"] == 0}
        orig_map = {it["original_task_id"]: it for it in orig_items}

        # Find common questions
        common = set(camo_v0.keys()) & set(orig_map.keys())
        if not common:
            results[model_name] = {"skip": "no overlapping questions"}
            continue

        camo_correct = 0
        orig_correct = 0
        by_task = defaultdict(lambda: {"camo_correct": 0, "orig_correct": 0, "total": 0})

        for qid in common:
            c = camo_v0[qid]
            o = orig_map[qid]
            v3 = v3_lookup.get(qid, {})
            task_type = v3.get("task", "?")

            by_task[task_type]["total"] += 1
            if c["expectation"] in c["actual_modalities"]:
                camo_correct += 1
                by_task[task_type]["camo_correct"] += 1
            if o["expectation"] in o["actual_modalities"]:
                orig_correct += 1
                by_task[task_type]["orig_correct"] += 1

        total = len(common)
        task_detail = {}
        for t in ["t2t", "ti2t", "t2i", "ti2i"]:
            d = by_task.get(t, {"camo_correct": 0, "orig_correct": 0, "total": 0})
            n = d["total"]
            task_detail[t] = {
                "orig_acc": d["orig_correct"] / n if n > 0 else None,
                "camo_acc": d["camo_correct"] / n if n > 0 else None,
                "delta": (d["camo_correct"] - d["orig_correct"]) / n if n > 0 else None,
                "n": n,
            }

        results[model_name] = {
            "total": total,
            "orig_acc": orig_correct / total if total > 0 else 0,
            "camo_acc": camo_correct / total if total > 0 else 0,
            "delta_acc": (camo_correct - orig_correct) / total if total > 0 else 0,
            "by_task": task_detail,
        }

    return results


# ──────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────

def print_table(title: str, headers: List[str], rows: List[List[str]]):
    """Print a formatted table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))


def fmt(val, decimals=4):
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def print_exp1(results: Dict):
    # Overall table
    headers = ["Model", "FPR", "TPR", "AUROC"]
    rows = []
    for name, r in results.items():
        roc = r.get("all") or {}
        rows.append([
            name,
            fmt(roc.get("fpr")),
            fmt(roc.get("tpr")),
            fmt(roc.get("auroc")),
        ])
    print_table("Exp 1: Modality Decision — Overall", headers, rows)

    # By input modality
    headers2 = ["Model", "Input", "FPR", "TPR", "AUROC"]
    rows2 = []
    for name, r in results.items():
        for key, label in [("text_input", "T-in"), ("image_input", "I-in")]:
            roc = r.get(key) or {}
            rows2.append([
                name, label,
                fmt(roc.get("fpr")),
                fmt(roc.get("tpr")),
                fmt(roc.get("auroc")),
            ])
    print_table("Exp 1: Modality Decision — By Input Modality", headers2, rows2)

    # Supplementary accuracy
    headers3 = ["Model", "Acc", "t2t", "ti2t", "t2i", "ti2i"]
    rows3 = []
    for name, r in results.items():
        rows3.append([
            name,
            fmt(r["overall_acc"]),
            fmt(r["task_acc"].get("t2t")),
            fmt(r["task_acc"].get("ti2t")),
            fmt(r["task_acc"].get("t2i")),
            fmt(r["task_acc"].get("ti2i")),
        ])
    print_table("Exp 1: Modality Decision — Accuracy (Supplementary)", headers3, rows3)


def print_exp2(results: Dict):
    headers = ["Model", "Consistency", "Flip Rate", "Mean Entropy", "t2t", "ti2t", "t2i", "ti2i"]
    rows = []
    for name, r in results.items():
        if "skip" in r:
            rows.append([name, r["skip"]] + ["-"] * 6)
            continue
        tc = r.get("task_consistency", {})
        rows.append([
            name,
            fmt(r["consistency_rate"]),
            fmt(r["flip_rate"]),
            fmt(r["mean_entropy"]),
            fmt(tc.get("t2t")),
            fmt(tc.get("ti2t")),
            fmt(tc.get("t2i")),
            fmt(tc.get("ti2i")),
        ])
    print_table("Exp 2: Prompt Variant Sensitivity (Consistency Rate)", headers, rows)


def print_exp3(results: Dict):
    headers = ["Model", "Task", "Metric", "Orig", "CAMO", "Δ"]
    rows = []
    for name, r in results.items():
        if "skip" in r:
            rows.append([name, "-", r["skip"], "-", "-", "-"])
            continue
        for t in ["t2t", "ti2t", "t2i", "ti2i"]:
            d = r.get("by_task", {}).get(t, {})
            rows.append([
                name, t,
                d.get("metric", "-"),
                fmt(d.get("orig_acc")),
                fmt(d.get("camo_acc")),
                fmt(d.get("delta")),
            ])
    print_table("Exp 3: Output Quality — Original vs CAMO", headers, rows)


def print_exp4(results: Dict):
    print(f"\n{'='*80}")
    print(f"  Exp 4: Surface-Form Cue Analysis (Original vs CAMO)")
    print(f"{'='*80}")

    headers = ["Model", "Orig Acc", "CAMO Acc", "Δ Acc"]
    rows = []
    for name, r in results.items():
        if "skip" in r:
            rows.append([name, r["skip"], "-", "-"])
            continue
        rows.append([
            name,
            fmt(r["orig_acc"]),
            fmt(r["camo_acc"]),
            fmt(r["delta_acc"]),
        ])
    print_table("Exp 4: Overall", headers, rows)

    # Per-task breakdown
    headers2 = ["Model", "Task", "Orig Acc", "CAMO Acc", "Δ", "N"]
    rows2 = []
    for name, r in results.items():
        if "skip" in r:
            continue
        for t in ["t2t", "ti2t", "t2i", "ti2i"]:
            d = r.get("by_task", {}).get(t, {})
            rows2.append([
                name, t,
                fmt(d.get("orig_acc")),
                fmt(d.get("camo_acc")),
                fmt(d.get("delta")),
                str(d.get("n", 0)),
            ])
    print_table("Exp 4: Per-Task Breakdown", headers2, rows2)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CAMO-Bench experiment analysis")
    p.add_argument("--v3", required=True, help="Path to v3.json")
    p.add_argument(
        "--models", required=True,
        help="CAMO results: 'name:path,name:path,...'",
    )
    p.add_argument(
        "--models_orig", default="",
        help="Original prompt results: 'name:path,name:path,...' (for Exp 4)",
    )
    p.add_argument(
        "--mock", nargs="*", default=[],
        help="Mock models: 'name' (generates synthetic results for external-control models)",
    )
    p.add_argument("--out", default=None, help="Output directory for JSON results")
    args = p.parse_args()

    # Load v3
    print("[info] Loading v3.json ...")
    v3_lookup = load_v3(args.v3)
    print(f"  {len(v3_lookup)} questions loaded")

    # Load CAMO model results
    model_dirs = parse_models_arg(args.models)
    model_results = {}
    for name, path in model_dirs.items():
        print(f"[info] Loading {name} from {path} ...")
        items = load_model_results(path, v3_lookup=v3_lookup)
        print(f"  {len(items)} items loaded")
        model_results[name] = items

    # Add mock models
    for mock_name in args.mock:
        print(f"[info] Generating mock results for {mock_name} ...")
        items = generate_mock_results(v3_lookup, mode=mock_name)
        print(f"  {len(items)} mock items generated")
        model_results[mock_name] = items

    # Load original prompt results (if provided)
    model_results_orig = {}
    if args.models_orig:
        orig_dirs = parse_models_arg(args.models_orig)
        for name, path in orig_dirs.items():
            print(f"[info] Loading {name} (original) from {path} ...")
            items = load_model_results(path, v3_lookup=v3_lookup)
            print(f"  {len(items)} items loaded")
            model_results_orig[name] = items

    # ── Run experiments ──
    print("\n" + "#" * 80)
    print("  CAMO-Bench Experiment Results")
    print("#" * 80)

    # Exp 1
    exp1 = exp1_modality_accuracy(model_results, v3_lookup)
    print_exp1(exp1)

    # Exp 2
    exp2 = exp2_variant_sensitivity(model_results, v3_lookup)
    print_exp2(exp2)

    # Exp 3
    if model_results_orig:
        exp3 = exp3_quality_comparison(model_results, model_results_orig, v3_lookup)
        print_exp3(exp3)
    else:
        print("\n[skip] Exp 3: No --models_orig provided")
        exp3 = {}

    # Exp 4
    if model_results_orig:
        exp4 = exp4_cue_analysis(model_results, model_results_orig, v3_lookup)
        print_exp4(exp4)
    else:
        print("\n[skip] Exp 4: No --models_orig provided")
        exp4 = {}

    # Save JSON
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        all_results = {
            "exp1_modality_accuracy": exp1,
            "exp2_variant_sensitivity": exp2,
            "exp3_quality_sensitivity": exp3,
            "exp4_cue_analysis": exp4,
        }
        out_path = out_dir / "experiment_results.json"
        out_path.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
