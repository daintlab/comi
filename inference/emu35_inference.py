#!/usr/bin/env python3
"""Emu3.5 free-choice inference over a COMI-Bench-format input file.

No modality flag is passed -- Emu3.5 decides text vs. image on its own via
its native BOI (`<|image start|>`) emission, using its own official
sampling settings throughout (not a forced single-modality path): 512px
resolution (`target_height=32, target_width=32` -- the model's own default
is to pick a resolution autonomously, usually 1024px/4096 image tokens,
which runs ~3.7-3.9x slower per sample with no measurable effect on
GenEval-style object/composition judgments; see the project's own
`experiment_512px.py` benchmark), `cfg_scale=3.0`,
`text_top_k=1024/text_top_p=0.9/text_temperature=1.0`,
`image_top_k=10240/image_top_p=1.0/image_temperature=1.0`,
`do_sample=True`, `max_new_tokens=5120`.

Requires a local clone of Emu3.5 (https://github.com/baaivision/Emu3.5)
importable as `src` -- point `--emu_repo` at it (this script `os.chdir`s
into it, since Emu3.5's own tokenizer config path is relative to its repo
root).

Input/output format: same contract as `anole_inference.py` -- see its
docstring. `other_parts` in the output (chain-of-thought segments Emu3.5
can interleave between images, e.g. `global_cot`/`image_cot`) is kept
alongside `text_output` (the non-CoT text segments only) rather than
merged into it, matching what this project's analysis scripts expect.

Usage:
    python emu35_inference.py --emu_repo /path/to/Emu3.5 \
        --input ../data/stage3_2variant_final.json \
        --out_dir outputs/emu35_2variant --image_dir outputs/emu35_2variant/images
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

MODEL_PATH = "BAAI/Emu3.5"
VQ_PATH = "BAAI/Emu3.5-VisionTokenizer"
TOKENIZER_PATH = "./src/tokenizer_emu3_ibq"
VQ_TYPE = "ibq"

SPECIAL_TOKENS = dict(
    BOS="<|extra_203|>", EOS="<|extra_204|>", PAD="<|endoftext|>",
    EOL="<|extra_200|>", EOF="<|extra_201|>", TMS="<|extra_202|>",
    IMG="<|image token|>", BOI="<|image start|>", EOI="<|image end|>",
    BSS="<|extra_100|>", ESS="<|extra_101|>",
    BOG="<|extra_60|>", EOG="<|extra_61|>", BOC="<|extra_50|>", EOC="<|extra_51|>",
)

TMPL_NOIMG = "<|extra_203|>You are a helpful assistant. USER: {question} ASSISTANT: <|extra_100|>"
UNC_PROMPT_NOIMG = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
TMPL_IMG = "<|extra_203|>You are a helpful assistant. USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>"
UNC_PROMPT_IMG = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"


class Cfg:
    pass


def build_cfg(max_new_tokens: int, cfg_scale: float, image_area: int, seed: int = 6666):
    cfg = Cfg()
    cfg.image_area = image_area
    cfg.classifier_free_guidance = cfg_scale
    cfg.unconditional_type = "no_text"
    cfg.streaming = False
    cfg.seed = seed
    cfg.special_tokens = SPECIAL_TOKENS
    cfg.target_height = 32
    cfg.target_width = 32  # 512px (32*16); see module docstring
    sampling_params = dict(
        use_cache=True,
        text_top_k=1024, text_top_p=0.9, text_temperature=1.0,
        image_top_k=10240, image_top_p=1.0, image_temperature=1.0,
        top_k=131072, top_p=1.0, temperature=1.0,
        num_beams_per_group=1, num_beam_groups=1, diversity_penalty=0.0,
        max_new_tokens=max_new_tokens, guidance_scale=1.0,
        use_differential_sampling=True,
    )
    sampling_params["do_sample"] = sampling_params["num_beam_groups"] <= 1
    sampling_params["num_beams"] = sampling_params["num_beams_per_group"] * sampling_params["num_beam_groups"]
    cfg.sampling_params = sampling_params
    return cfg


def run_one(model, tokenizer, vq_model, cfg, emu_generate, multimodal_decode, build_image,
            prompt_text: str, image_abspath: str | None, out_image_prefix: str | None):
    if image_abspath:
        image = Image.open(image_abspath).convert("RGB")
        image_str = build_image(image, cfg, tokenizer, vq_model)
        prompt = TMPL_IMG.format(question=prompt_text).replace("<|IMAGE|>", image_str)
        unc_prompt = UNC_PROMPT_IMG.replace("<|IMAGE|>", image_str)
    else:
        prompt = TMPL_NOIMG.format(question=prompt_text)
        unc_prompt = UNC_PROMPT_NOIMG

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
        bos = torch.tensor([[cfg.special_token_ids["BOS"]]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([bos, input_ids], dim=1)
    unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    gen_ids = None
    with torch.inference_mode():
        for result_tokens in emu_generate(cfg, model, tokenizer, input_ids, unconditional_ids,
                                           full_unconditional_ids=None, force_same_image_size=True):
            gen_ids = result_tokens
    gen_ids = gen_ids.tolist() if hasattr(gen_ids, "tolist") else list(gen_ids)

    boi_id = cfg.special_token_ids["BOI"]
    pred_modality = "image" if boi_id in gen_ids else "text"

    full_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    segments = multimodal_decode(full_text, tokenizer, vq_model)
    text_parts, images, other_parts = [], [], []
    for seg_type, seg_val in segments:
        if seg_type == "image":
            images.append(seg_val)
        elif seg_type == "text":
            if seg_val.strip():
                text_parts.append(seg_val.strip())
        else:
            other_parts.append((seg_type, seg_val.strip()))

    image_paths = []
    for i, img in enumerate(images):
        p = f"{out_image_prefix}_img{i}.png"
        img.save(p)
        image_paths.append(p)

    return {
        "pred_modality": pred_modality,
        "n_generated_tokens": len(gen_ids),
        "text_output": "\n".join(text_parts),
        "other_parts": other_parts,
        "generated_image_paths": image_paths,
    }


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
    ap.add_argument("--emu_repo", required=True, help="path to a local Emu3.5 clone")
    ap.add_argument("--input", required=True, help="COMI-Bench-format JSON")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_dir", default=None, help="default: <out_dir>/images")
    ap.add_argument("--max_new_tokens", type=int, default=5120)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--image_area", type=int, default=518400)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    image_dir = args.image_dir or os.path.join(args.out_dir, "images")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    emu_repo = os.path.abspath(args.emu_repo)
    sys.path.insert(0, emu_repo)
    os.chdir(emu_repo)  # tokenizer_path in configs is relative to repo root

    from src.utils.model_utils import build_emu3p5  # noqa: E402
    from src.utils.generation_utils import generate as emu_generate, multimodal_decode  # noqa: E402
    from src.utils.input_utils import build_image  # noqa: E402

    model, tokenizer, vq_model = build_emu3p5(
        MODEL_PATH, TOKENIZER_PATH, VQ_PATH, vq_type=VQ_TYPE, model_device=0, vq_device="cuda:0",
    )
    cfg = build_cfg(args.max_new_tokens, args.cfg_scale, args.image_area)
    cfg.special_token_ids = {k: tokenizer.encode(v)[0] for k, v in SPECIAL_TOKENS.items()}

    data = json.loads(Path(args.input).read_text())
    items = list(iter_items(data))
    if args.limit:
        items = items[: args.limit]

    out_path = os.path.join(args.out_dir, "results.jsonl")
    with open(out_path, "w") as out_f:
        for i, (variant_id, task_id, source, gold_modality, prompt_text, image_path) in enumerate(items):
            out_prefix = os.path.join(image_dir, variant_id)
            result = run_one(model, tokenizer, vq_model, cfg, emu_generate, multimodal_decode,
                              build_image, prompt_text, image_path, out_prefix)
            rec = {"variant_id": variant_id, "task_id": task_id, "source": source,
                   "gold_modality": gold_modality, **result}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"\r  {i + 1}/{len(items)}", end="", file=sys.stderr)
    print(f"\nsaved -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
