#!/usr/bin/env python3
"""Anole-7B free-choice inference over a COMI-Bench-format input file.

No modality flag is passed -- the model decides text vs. image on its own
via Chameleon's `Options()` (its official defaults: text greedy=False,
temp=1.0, top_p=0.9, repetition_penalty=1.2; image greedy=False, temp=0.7,
top_p=0.9, cfg.guidance_scale_text=3.0, cfg.guidance_scale_image=1.2 --
both held simultaneously, so no special-casing is needed for free choice).
This mirrors the official repo's own single-modality demo scripts
(`text2image.py`, `scripts/run_samples_t2t.py`), just without forcing either
`options.txt` or `options.img` to False.

Requires a local clone of Anole / Chameleon (the official repo this project
evaluated: https://github.com/GAIR-NLP/anole) importable as `chameleon` --
point `--anole_repo` at it, or set it on PYTHONPATH yourself.

Input: a COMI-Bench-format JSON (`data/stage3_2variant_final.json` or
`data/comi_bench_final_original_5682.json`), `{"results": [...]}` where each
result has either `variants: [{text, slot_assigned}, ...]` (2-variant file)
or a single `prompt` (original-prompt file). `image_path` (optional) is
loaded as the input image for TI2T/TI2I items.

Output: one JSON object per line, matching what `evaluation/compute_msa_tp.py`
expects: {variant_id, task_id, source, gold_modality, pred_modality,
text_output, generated_image_paths}.

Usage:
    python anole_inference.py --anole_repo /path/to/anole \
        --input ../data/stage3_2variant_final.json \
        --out_dir outputs/anole_2variant --image_dir outputs/anole_2variant/images
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SEED = 42
IMAGE_TOKEN_COUNT = 1024


def build_model(anole_repo: str, ckpt_path: str):
    sys.path.insert(0, anole_repo)
    from chameleon.inference.chameleon import ChameleonInferenceModel  # noqa: E402
    from constants import (  # noqa: E402
        TOKENIZER_IMAGE_CFG_PATH, TOKENIZER_IMAGE_PATH, TOKENIZER_TEXT_PATH,
    )
    return ChameleonInferenceModel(
        ckpt_path, TOKENIZER_TEXT_PATH.as_posix(),
        TOKENIZER_IMAGE_CFG_PATH.as_posix(), TOKENIZER_IMAGE_PATH.as_posix(),
    )


def run_one(model, Options, split_token_sequence, prompt_text: str,
            image_abspath: str | None, out_image_path: str | None):
    prompt_ui = []
    if image_abspath is not None:
        prompt_ui.append({"type": "image", "value": f"file:{image_abspath}"})
    prompt_ui.append({"type": "text", "value": prompt_text})

    options = Options(seed=SEED)
    tokens = model.generate(batch_prompt_ui=[prompt_ui], options=options)
    boi, eoi = model.vocab.begin_image, model.vocab.end_image
    segments = split_token_sequence(tokens, boi, eoi)

    text_parts, image_paths = [], []
    for seg_type, seg_tokens in segments:
        if seg_type == "text_seg":
            decoded = model.decode_text(seg_tokens)[0].strip()
            if decoded:
                text_parts.append(decoded)
        elif seg_type == "image_seg" and seg_tokens.shape[1] == IMAGE_TOKEN_COUNT:
            img = model.decode_image(seg_tokens)[0]
            if out_image_path:
                img.save(out_image_path)
                image_paths.append(out_image_path)

    pred_modality = "image" if any(s[0] == "image_seg" for s in segments) else "text"
    return {
        "pred_modality": pred_modality,
        "n_generated_tokens": int(tokens.shape[1]),
        "text_output": "\n".join(text_parts),
        "generated_image_paths": image_paths,
    }


def iter_items(data: dict):
    """Yield (variant_id, task_id, source, gold_modality, prompt_text, image_path)
    for every prompt in a 2-variant or original-prompt COMI-Bench file."""
    for r in data["results"]:
        task_id = r["id"]
        source = r["source"]
        gold_modality = r.get("gt_modality") or r.get("gold_modality")
        image_path = r.get("image_path")
        variants = r.get("variants")
        if variants:
            for v in variants:
                slot = v.get("slot_assigned", "v")
                yield f"{task_id}__{slot}", task_id, source, gold_modality, v["text"], image_path
        else:
            yield f"{task_id}__orig", task_id, source, gold_modality, r["prompt"], image_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anole_repo", required=True, help="path to a local Anole/Chameleon clone")
    ap.add_argument("--ckpt_path", required=True, help="path to the Anole-7b-v0.1 checkpoint")
    ap.add_argument("--input", required=True, help="COMI-Bench-format JSON")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_dir", default=None, help="default: <out_dir>/images")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    image_dir = args.image_dir or os.path.join(args.out_dir, "images")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    sys.path.insert(0, args.anole_repo)
    from chameleon.inference.chameleon import Options  # noqa: E402
    from inference import split_token_sequence  # noqa: E402  (official repo's own helper)

    model = build_model(args.anole_repo, args.ckpt_path)

    data = json.loads(Path(args.input).read_text())
    items = list(iter_items(data))
    if args.limit:
        items = items[: args.limit]

    out_path = os.path.join(args.out_dir, "results.jsonl")
    with open(out_path, "w") as out_f:
        for i, (variant_id, task_id, source, gold_modality, prompt_text, image_path) in enumerate(items):
            out_image_path = os.path.join(image_dir, f"{variant_id}.png")
            result = run_one(model, Options, split_token_sequence, prompt_text, image_path, out_image_path)
            rec = {"variant_id": variant_id, "task_id": task_id, "source": source,
                   "gold_modality": gold_modality, **result}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"\r  {i + 1}/{len(items)}", end="", file=sys.stderr)
    print(f"\nsaved -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
