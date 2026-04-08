#!/usr/bin/env python3
"""Collect TI2T datasets: VQAv2, ScienceQA."""

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


def collect_vqav2():
    """Collect VQAv2 (lmms-lab/VQAv2), filtered by answer_type == 'other'."""
    out_dir = os.path.join(config.TI2T_DIR, "vqav2")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    for split in ["validation", "test"]:
        out_path = os.path.join(out_dir, f"{split}.jsonl")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists")
            continue

        print(f"[vqav2] Loading split={split} ...")
        try:
            ds = load_dataset("lmms-lab/VQAv2", split=split)
        except Exception as e:
            print(f"[vqav2] Could not load split={split}: {e}")
            continue

        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc=f"vqav2/{split}"):
                if row.get("answer_type") != "other":
                    continue

                # Save image
                img = row.get("image")
                qid = row.get("question_id", count)
                img_filename = f"{split}_{qid}.png"
                img_path = os.path.join(img_dir, img_filename)
                if img is not None:
                    _save_image(img, img_path)

                record = {
                    k: v for k, v in row.items() if k != "image"
                }
                record["image_path"] = os.path.join("images", img_filename)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        print(f"[vqav2] {split}: {count} samples (answer_type=='other') -> {out_path}")

    return out_dir


def collect_scienceqa():
    """Collect ScienceQA (derek-thomas/ScienceQA), filtered by image!=None and subject/topic."""
    out_dir = os.path.join(config.TI2T_DIR, "scienceqa")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    allowed = set()
    for topics in config.SCIENCEQA_SUBJECTS.values():
        for t in topics:
            allowed.add(t.lower())

    for split in ["train", "validation", "test"]:
        out_path = os.path.join(out_dir, f"{split}.jsonl")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists")
            continue

        print(f"[scienceqa] Loading split={split} ...")
        try:
            ds = load_dataset("derek-thomas/ScienceQA", split=split)
        except Exception as e:
            print(f"[scienceqa] Could not load split={split}: {e}")
            continue

        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for idx, row in enumerate(tqdm(ds, desc=f"scienceqa/{split}")):
                # Filter: must have image
                img = row.get("image")
                if img is None:
                    continue

                # Filter: subject/topic
                topic = (row.get("topic") or "").lower()
                if topic not in allowed:
                    continue

                # Save image
                img_filename = f"{split}_{idx:06d}.png"
                img_path = os.path.join(img_dir, img_filename)
                _save_image(img, img_path)

                record = {k: v for k, v in row.items() if k != "image"}
                record["image_path"] = os.path.join("images", img_filename)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        print(f"[scienceqa] {split}: {count} samples -> {out_path}")

    return out_dir


def main():
    print("=" * 60)
    print("Collecting TI2T datasets")
    print("=" * 60)

    vqav2_dir = collect_vqav2()
    scienceqa_dir = collect_scienceqa()

    # Summary
    print("\n" + "=" * 60)
    print("TI2T Collection Summary")
    print("=" * 60)
    for name, d in [("VQAv2", vqav2_dir), ("ScienceQA", scienceqa_dir)]:
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".jsonl"):
                n = sum(1 for _ in open(os.path.join(d, fname)))
                print(f"  {name}/{fname}: {n} samples")
        n_imgs = len([f for f in os.listdir(os.path.join(d, "images")) if f.endswith(".png")])
        print(f"  {name}/images: {n_imgs} files")


if __name__ == "__main__":
    main()
