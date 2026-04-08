#!/usr/bin/env python3
"""Collect T2T datasets: GSM8K, MMLU."""

import json
import os
import sys

from datasets import load_dataset
from tqdm import tqdm

import config


def collect_gsm8k():
    """Collect GSM8K (openai/gsm8k, config=main)."""
    out_dir = os.path.join(config.T2T_DIR, "gsm8k")
    os.makedirs(out_dir, exist_ok=True)

    for split in ["train", "test"]:
        out_path = os.path.join(out_dir, f"{split}.jsonl")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists")
            continue

        print(f"[gsm8k] Loading split={split} ...")
        ds = load_dataset("openai/gsm8k", "main", split=split)

        with open(out_path, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc=f"gsm8k/{split}"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[gsm8k] {split}: {len(ds)} samples -> {out_path}")

    return out_dir


def collect_mmlu():
    """Collect MMLU (cais/mmlu, config=all) filtered by subjects."""
    out_dir = os.path.join(config.T2T_DIR, "mmlu")
    os.makedirs(out_dir, exist_ok=True)

    for split in ["test", "validation", "dev"]:
        out_path = os.path.join(out_dir, f"{split}.jsonl")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists")
            continue

        print(f"[mmlu] Loading split={split} ...")
        try:
            ds = load_dataset("cais/mmlu", "all", split=split)
        except Exception as e:
            print(f"[mmlu] Could not load split={split}: {e}")
            continue

        subjects_set = set(config.MMLU_SUBJECTS)
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc=f"mmlu/{split}"):
                if row.get("subject") in subjects_set:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1

        print(f"[mmlu] {split}: {count}/{len(ds)} samples (filtered) -> {out_path}")

    return out_dir


def main():
    print("=" * 60)
    print("Collecting T2T datasets")
    print("=" * 60)

    gsm8k_dir = collect_gsm8k()
    mmlu_dir = collect_mmlu()

    # Summary
    print("\n" + "=" * 60)
    print("T2T Collection Summary")
    print("=" * 60)
    for name, d in [("GSM8K", gsm8k_dir), ("MMLU", mmlu_dir)]:
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".jsonl"):
                n = sum(1 for _ in open(os.path.join(d, fname)))
                print(f"  {name}/{fname}: {n} samples")


if __name__ == "__main__":
    main()
