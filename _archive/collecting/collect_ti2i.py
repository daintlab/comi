#!/usr/bin/env python3
"""Collect TI2I datasets: MagicBrush, ImgEditBench."""

import json
import os

from datasets import load_dataset
from tqdm import tqdm

import config


def _save_image(img, path):
    """Save a PIL image if not already saved."""
    if os.path.exists(path):
        return
    img.save(path)


def collect_magicbrush():
    """Collect MagicBrush (osunlp/MagicBrush): single-turn (turn_index==1), mask-free."""
    out_dir = os.path.join(config.TI2I_DIR, "magicbrush")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    for split in ["dev"]:
        out_path = os.path.join(out_dir, f"{split}.jsonl")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists")
            continue

        print(f"[magicbrush] Loading split={split} ...")
        try:
            ds = load_dataset("osunlp/MagicBrush", split=split)
        except Exception as e:
            print(f"[magicbrush] Could not load split={split}: {e}")
            continue

        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for idx, row in enumerate(tqdm(ds, desc=f"magicbrush/{split}")):
                # Filter: single-turn only
                if row.get("turn_index") != 1:
                    continue

                # Save source and target images
                record = {}
                for key in row:
                    if key in ("source_img", "target_img", "mask_img"):
                        continue
                    record[key] = row[key]

                src_img = row.get("source_img")
                tgt_img = row.get("target_img")
                img_id = row.get("img_id", f"{split}_{idx:06d}")

                if src_img is not None:
                    src_name = f"{img_id}_source.png"
                    _save_image(src_img, os.path.join(img_dir, src_name))
                    record["source_img_path"] = os.path.join("images", src_name)

                if tgt_img is not None:
                    tgt_name = f"{img_id}_target.png"
                    _save_image(tgt_img, os.path.join(img_dir, tgt_name))
                    record["target_img_path"] = os.path.join("images", tgt_name)

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        print(f"[magicbrush] {split}: {count} samples (single-turn, mask-free) -> {out_path}")

    return out_dir


def collect_imgeditbench():
    """Collect ImgEditBench (PKU-YuanGroup/ImgEdit, benchmark split): single-turn, selected subtasks."""
    out_dir = os.path.join(config.TI2I_DIR, "imgeditbench")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "benchmark.jsonl")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        return out_dir

    print("[imgeditbench] Loading benchmark split ...")
    try:
        ds = load_dataset("PKU-YuanGroup/ImgEdit", split="benchmark")
    except Exception as e:
        print(f"[imgeditbench] Could not load: {e}")
        print("[imgeditbench] Trying with config name 'benchmark' ...")
        try:
            ds = load_dataset("PKU-YuanGroup/ImgEdit", "benchmark", split="train")
        except Exception as e2:
            print(f"[imgeditbench] Also failed: {e2}")
            return out_dir

    subtasks_set = set(config.IMGEDIT_SUBTASKS)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(tqdm(ds, desc="imgeditbench")):
            # Filter: subtask
            subtask = (row.get("subtask") or row.get("category") or "").lower()
            if subtask not in subtasks_set:
                continue

            record = {}
            for key in row:
                val = row[key]
                if hasattr(val, "save"):  # PIL Image
                    img_name = f"{idx:06d}_{key}.png"
                    _save_image(val, os.path.join(img_dir, img_name))
                    record[f"{key}_path"] = os.path.join("images", img_name)
                else:
                    record[key] = val

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"[imgeditbench] {count} samples (filtered subtasks) -> {out_path}")
    return out_dir


def main():
    print("=" * 60)
    print("Collecting TI2I datasets")
    print("=" * 60)

    mb_dir = collect_magicbrush()
    ie_dir = collect_imgeditbench()

    # Summary
    print("\n" + "=" * 60)
    print("TI2I Collection Summary")
    print("=" * 60)
    for name, d in [("MagicBrush", mb_dir), ("ImgEditBench", ie_dir)]:
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".jsonl"):
                n = sum(1 for _ in open(os.path.join(d, fname)))
                print(f"  {name}/{fname}: {n} samples")
        img_d = os.path.join(d, "images")
        if os.path.isdir(img_d):
            n_imgs = len(os.listdir(img_d))
            print(f"  {name}/images: {n_imgs} files")


if __name__ == "__main__":
    main()
