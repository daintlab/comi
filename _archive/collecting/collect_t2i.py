#!/usr/bin/env python3
"""Collect T2I datasets: GenEval, DPG-Bench."""

import json
import os

from datasets import load_dataset
from tqdm import tqdm

import config


def collect_geneval():
    """Collect GenEval (hku-nlp/GenEval). Falls back to GitHub JSON if HF fails."""
    out_dir = os.path.join(config.T2I_DIR, "geneval")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "prompts.jsonl")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        return out_dir

    print("[geneval] Loading from HuggingFace ...")
    try:
        ds = load_dataset("hku-nlp/GenEval", split="train")
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc="geneval"):
                record = dict(row)
                # Remove non-serializable fields if any
                for k, v in list(record.items()):
                    if hasattr(v, "save"):  # PIL Image
                        record.pop(k)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        print(f"[geneval] {count} prompts -> {out_path}")
    except Exception as e:
        print(f"[geneval] HF load failed: {e}")
        print("[geneval] Trying GitHub fallback ...")
        _geneval_github_fallback(out_path)

    return out_dir


def _geneval_github_fallback(out_path):
    """Download GenEval prompts from GitHub."""
    import urllib.request

    url = "https://raw.githubusercontent.com/djghosh13/geneval/main/prompts/evaluation_metadata.jsonl"
    print(f"[geneval] Downloading from {url} ...")
    try:
        resp = urllib.request.urlopen(url)
        data = resp.read().decode("utf-8")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(data)
        count = sum(1 for line in data.strip().split("\n") if line.strip())
        print(f"[geneval] {count} prompts (GitHub) -> {out_path}")
    except Exception as e2:
        print(f"[geneval] GitHub fallback also failed: {e2}")
        print("[geneval] Please download manually.")


def collect_dpg_bench():
    """Collect DPG-Bench (ELLA-Diffusion/DPG-Bench). Falls back to GitHub if HF fails."""
    out_dir = os.path.join(config.T2I_DIR, "dpg_bench")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "prompts.jsonl")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        return out_dir

    print("[dpg_bench] Loading from HuggingFace ...")
    try:
        ds = load_dataset("ELLA-Diffusion/DPG-Bench", split="train")
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc="dpg_bench"):
                record = dict(row)
                for k, v in list(record.items()):
                    if hasattr(v, "save"):  # PIL Image
                        record.pop(k)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        print(f"[dpg_bench] {count} prompts -> {out_path}")
    except Exception as e:
        print(f"[dpg_bench] HF load failed: {e}")
        print("[dpg_bench] Trying GitHub fallback ...")
        _dpg_bench_github_fallback(out_path)

    return out_dir


def _dpg_bench_github_fallback(out_path):
    """Download DPG-Bench prompts from GitHub."""
    import urllib.request

    url = "https://raw.githubusercontent.com/ELLA-Diffusion/DPG-Bench/main/prompts.csv"
    print(f"[dpg_bench] Downloading from {url} ...")
    try:
        import csv
        import io

        resp = urllib.request.urlopen(url)
        data = resp.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(data))
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for row in reader:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
                count += 1
        print(f"[dpg_bench] {count} prompts (GitHub) -> {out_path}")
    except Exception as e2:
        print(f"[dpg_bench] GitHub fallback also failed: {e2}")
        print("[dpg_bench] Please download manually.")


def main():
    print("=" * 60)
    print("Collecting T2I datasets")
    print("=" * 60)

    geneval_dir = collect_geneval()
    dpg_dir = collect_dpg_bench()

    # Summary
    print("\n" + "=" * 60)
    print("T2I Collection Summary")
    print("=" * 60)
    for name, d in [("GenEval", geneval_dir), ("DPG-Bench", dpg_dir)]:
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".jsonl"):
                n = sum(1 for _ in open(os.path.join(d, fname)))
                print(f"  {name}/{fname}: {n} prompts")


if __name__ == "__main__":
    main()
