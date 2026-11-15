#!/usr/bin/env python3
"""Tar-7B free-choice inference over a COMI-Bench-format input file.

No modality flag is passed -- the model decides text vs. image on its own.
Tar's official repo only ships single-modality decoding settings (text:
`i2t_inference.py` + checkpoint `generation_config.json`; image:
`t2i_inference.py`), and HF's `generate()` applies one decoding config per
call, so free choice is generated in two stages over one continuous
sequence rather than with a single call:

  Stage 1: official text settings (do_sample=True, temperature=0.7,
    top_p=0.8, top_k=20, repetition_penalty=1.05, max_new_tokens=256),
    with `<im_start>` added to `eos_token_id` so generation stops the
    instant the model chooses to start an image.
  Stage 2 (only if Stage 1 stopped on `<im_start>`): official image
    settings (do_sample=True, temperature=1.0, top_p=0.95, top_k=1200,
    max_new_tokens=730), continuing from Stage 1's KV cache rather than
    re-prefilling the whole sequence. HF's high-level `generate()` doesn't
    support resuming from an external `past_key_values` this way (its
    `cache_position` bookkeeping assumes a fresh call), so Stage 2 is a
    manual step loop using HF's own `TemperatureLogitsWarper` /
    `TopKLogitsWarper` / `TopPLogitsWarper` -- not a hand-rolled
    reimplementation of the sampling math, just HF's own classes called
    directly instead of through `generate()`.

Using a single set of decoding settings for both stages (e.g. the image
settings' large top_k/top_p applied to the text stage too) would change
which tokens -- including the text-vs-image decision itself -- get sampled
at the free-choice boundary, confounding the model's actual free-choice
behavior with an artifact of the decoding setting. The two stages must use
each one's own official settings for MSA under free choice to mean what it's
supposed to mean.

Requires a local clone of Tar (https://github.com/csuhan/Tar) importable as
`tar_unified` (its `TarUnified` wrapper) -- point `--tar_repo` at it.

Input/output format: same contract as `anole_inference.py` -- see its
docstring.

Usage:
    python tar_inference.py --tar_repo /path/to/Tar \
        --input ../data/stage3_2variant_final.json \
        --out_dir outputs/tar_2variant --image_dir outputs/tar_2variant/images
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

SEED = 42
TEXT_DECODING = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                      repetition_penalty=1.05, max_new_tokens=256)
IMAGE_DECODING = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=1200,
                       max_new_tokens=730)  # 1 scale marker + up to 729 image tokens


def generate_free_choice_two_stage(tar, prompt: str, image, cfg_scale: float):
    if image is not None:
        user_content = f"{tar.encode_image(image)}\n{prompt}"
    else:
        user_content = prompt
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content}]
    input_text = tar.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tar.tok(input_text, return_tensors="pt").input_ids.to(tar.device)
    prompt_len = input_ids.shape[1]

    torch.manual_seed(SEED)
    with torch.inference_mode():
        out1 = tar.model.generate(
            input_ids, eos_token_id=[tar.tok.eos_token_id, tar.BOI_ID],
            return_dict_in_generate=True, use_cache=True, **TEXT_DECODING,
        )
    gen1 = out1.sequences
    stage1_ids = gen1[0, prompt_len:].cpu().tolist()
    stopped_on_boi = bool(stage1_ids) and stage1_ids[-1] == tar.BOI_ID

    if not stopped_on_boi:
        full_ids = stage1_ids
    else:
        torch.manual_seed(SEED)
        warpers = LogitsProcessorList([
            TemperatureLogitsWarper(IMAGE_DECODING["temperature"]),
            TopKLogitsWarper(IMAGE_DECODING["top_k"]),
            TopPLogitsWarper(IMAGE_DECODING["top_p"]),
        ])
        past_kv = out1.past_key_values
        cur_token = gen1[:, -1:]
        new_ids = []
        stop_ids = {tar.tok.eos_token_id, tar.EOI_ID}
        with torch.inference_mode():
            for _ in range(IMAGE_DECODING["max_new_tokens"]):
                step_out = tar.model(input_ids=cur_token, past_key_values=past_kv, use_cache=True)
                logits = warpers(gen1, step_out.logits[:, -1, :])
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                tok_id = next_token.item()
                new_ids.append(tok_id)
                if tok_id in stop_ids:
                    break
                past_kv = step_out.past_key_values
                cur_token = next_token
        full_ids = stage1_ids + new_ids

    segments = tar.split_segments(full_ids, tar.BOI_ID, tar.EOI_ID)
    text_parts, images = [], []
    for stype, seg_ids in segments:
        if stype == "text":
            decoded = tar.tok.decode(seg_ids, skip_special_tokens=True).strip()
            if decoded:
                text_parts.append(decoded)
        else:
            scale, img = tar.decode_image_segment(seg_ids, cfg_scale=cfg_scale)
            if img is not None:
                images.append(img)

    pred_modality = "image" if any(s[0] == "image" for s in segments) else "text"
    return {"pred_modality": pred_modality, "n_generated_tokens": len(full_ids),
            "text_output": "\n".join(text_parts), "images": images}


def iter_items(data: dict):
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
    ap.add_argument("--tar_repo", required=True, help="path to a local Tar clone")
    ap.add_argument("--input", required=True, help="COMI-Bench-format JSON")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_dir", default=None, help="default: <out_dir>/images")
    ap.add_argument("--cfg_scale", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    image_dir = args.image_dir or os.path.join(args.out_dir, "images")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    sys.path.insert(0, args.tar_repo)
    sys.path.insert(0, os.path.join(args.tar_repo, "scripts"))
    from tar_unified import TarUnified  # noqa: E402

    tar = TarUnified()

    data = json.loads(Path(args.input).read_text())
    items = list(iter_items(data))
    if args.limit:
        items = items[: args.limit]

    out_path = os.path.join(args.out_dir, "results.jsonl")
    with open(out_path, "w") as out_f:
        for i, (variant_id, task_id, source, gold_modality, prompt_text, image_path) in enumerate(items):
            image = None
            if image_path:
                from PIL import Image
                image = Image.open(image_path).convert("RGB")
            result = generate_free_choice_two_stage(tar, prompt_text, image, args.cfg_scale)
            image_paths = []
            for j, img in enumerate(result.pop("images")):
                p = os.path.join(image_dir, f"{variant_id}_img{j}.png")
                img.save(p)
                image_paths.append(p)
            rec = {"variant_id": variant_id, "task_id": task_id, "source": source,
                   "gold_modality": gold_modality, "generated_image_paths": image_paths, **result}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"\r  {i + 1}/{len(items)}", end="", file=sys.stderr)
    print(f"\nsaved -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
