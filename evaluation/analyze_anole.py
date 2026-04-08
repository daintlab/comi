#!/usr/bin/env python3
"""Detailed Anole analysis: AUROC and accuracy breakdown by source and task type."""

import json
import glob
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

V3_PATH = "v3.json"
ANOLE_DIR = "/nas/home/ysk1207/cams-benchmark/anole/anole/outputs/v3"


def load_v3():
    data = json.load(open(V3_PATH, encoding="utf-8"))
    return {r["task_id"]: r for r in data["results"]}


def resolve_task_id(task_id):
    if task_id.endswith("_orig"):
        return task_id[:-5]
    if "_v" in task_id:
        parts = task_id.rsplit("_v", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
    return task_id


def load_anole(v3_lookup):
    items = []
    for d in sorted(glob.glob(os.path.join(ANOLE_DIR, "item_*"))):
        meta_f = os.path.join(d, "meta.json")
        out_f = os.path.join(d, "output_r00.json")
        if not os.path.exists(out_f):
            continue
        meta = json.load(open(meta_f, encoding="utf-8"))
        output = json.load(open(out_f, encoding="utf-8"))

        task_id = meta.get("task_id", "")
        orig_id = resolve_task_id(task_id)
        v3 = v3_lookup.get(orig_id, {})

        # variant idx
        variant_idx = 0
        if "_v" in task_id:
            parts = task_id.rsplit("_v", 1)
            if len(parts) == 2 and parts[1].isdigit():
                variant_idx = int(parts[1])

        boi_prob = None
        for s in output:
            if s.get("type") == "meta":
                boi_prob = s.get("boi_prob")

        has_text = any(s.get("type") == "text" and s.get("content", "").strip() for s in output)
        has_image = any(s.get("type") == "image" for s in output)
        actual_mods = set()
        if has_text:
            actual_mods.add("text")
        if has_image:
            actual_mods.add("image")

        expect = meta.get("expectation", "")
        if expect not in ("text", "image"):
            expect = v3.get("gold_modality", expect)

        items.append({
            "task_id": task_id,
            "orig_id": orig_id,
            "variant_idx": variant_idx,
            "source": v3.get("source", meta.get("source", "?")),
            "task": v3.get("task", "?"),
            "expectation": expect,
            "actual_modalities": actual_mods,
            "boi_prob": boi_prob,
        })
    return items


def compute_stats(items):
    """Compute accuracy, FPR, TPR, AUROC for a group of items."""
    total = len(items)
    correct = sum(1 for it in items if it["expectation"] in it["actual_modalities"])
    acc = correct / total if total > 0 else None

    probs = []
    labels = []
    for it in items:
        if it["boi_prob"] is not None:
            probs.append(it["boi_prob"])
            labels.append(1 if it["expectation"] == "image" else 0)

    auroc = fpr = tpr = None
    if len(set(labels)) == 2:
        auroc = roc_auc_score(labels, probs)
        fpr_arr, tpr_arr, thresholds = roc_curve(labels, probs)
        j = tpr_arr - fpr_arr
        idx = np.argmax(j)
        fpr = float(fpr_arr[idx])
        tpr = float(tpr_arr[idx])

    return {
        "n": total,
        "correct": correct,
        "acc": acc,
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
    }


def fmt(v, d=4):
    if v is None:
        return "-"
    return f"{v:.{d}f}"


def print_table(title, headers, rows):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    col_w = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print("-+-".join("-" * w for w in col_w))
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, col_w)))


def main():
    v3_lookup = load_v3()
    items = load_anole(v3_lookup)

    # Use variant 0 only
    v0 = [it for it in items if it["variant_idx"] == 0]
    print(f"Total items (v0): {len(v0)}")

    # --- By source ---
    by_source = defaultdict(list)
    for it in v0:
        by_source[it["source"]].append(it)

    headers = ["Source", "N", "Acc", "FPR", "TPR", "AUROC"]
    rows = []
    for src in sorted(by_source.keys()):
        s = compute_stats(by_source[src])
        rows.append([src, str(s["n"]), fmt(s["acc"]), fmt(s["fpr"]), fmt(s["tpr"]), fmt(s["auroc"])])
    print_table("Anole — By Source (variant 0)", headers, rows)

    # --- By task type ---
    by_task = defaultdict(list)
    for it in v0:
        by_task[it["task"]].append(it)

    rows2 = []
    for t in ["t2t", "ti2t", "t2i", "ti2i"]:
        s = compute_stats(by_task.get(t, []))
        rows2.append([t, str(s["n"]), fmt(s["acc"]), fmt(s["fpr"]), fmt(s["tpr"]), fmt(s["auroc"])])
    print_table("Anole — By Task Type (variant 0)", headers, rows2)

    # --- By source x task type ---
    by_src_task = defaultdict(list)
    for it in v0:
        by_src_task[(it["source"], it["task"])].append(it)

    headers3 = ["Source", "Task", "N", "Expect", "Acc", "Mean boi_prob"]
    rows3 = []
    for (src, task) in sorted(by_src_task.keys()):
        group = by_src_task[(src, task)]
        n = len(group)
        correct = sum(1 for it in group if it["expectation"] in it["actual_modalities"])
        acc = correct / n if n > 0 else 0
        probs = [it["boi_prob"] for it in group if it["boi_prob"] is not None]
        mean_prob = np.mean(probs) if probs else 0
        expect = group[0]["expectation"] if group else "?"
        rows3.append([src, task, str(n), expect, fmt(acc), fmt(mean_prob, 4)])
    print_table("Anole — By Source x Task (variant 0)", headers3, rows3)

    # --- boi_prob distribution for image-expected tasks by source ---
    print(f"\n{'='*90}")
    print(f"  Anole — boi_prob distribution for image-expected tasks")
    print(f"{'='*90}")
    img_items = [it for it in v0 if it["expectation"] == "image"]
    by_src_img = defaultdict(list)
    for it in img_items:
        by_src_img[it["source"]].append(it)

    headers4 = ["Source", "N", "Mean", "Median", "Min", "Max", "Acc"]
    rows4 = []
    for src in sorted(by_src_img.keys()):
        group = by_src_img[src]
        probs = [it["boi_prob"] for it in group if it["boi_prob"] is not None]
        correct = sum(1 for it in group if "image" in it["actual_modalities"])
        acc = correct / len(group)
        if probs:
            rows4.append([
                src, str(len(group)),
                fmt(np.mean(probs)), fmt(np.median(probs)),
                fmt(np.min(probs)), fmt(np.max(probs)),
                fmt(acc),
            ])
    print(" | ".join(h.ljust(12) for h in headers4))
    print("-+-".join("-" * 12 for _ in headers4))
    for row in rows4:
        print(" | ".join(str(v).ljust(12) for v in row))


if __name__ == "__main__":
    main()
