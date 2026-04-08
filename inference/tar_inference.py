#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-load batch runner for Tar-7B multimodal inference.

The model decides the output modality (text vs image) autonomously:
- If the model generates <im_start> token → image generation mode
- Otherwise → text generation mode

This mirrors Anole/Chameleon's approach where the model chooses
whether to output text or images based on the prompt alone.

Input JSON format (compatible with aug_bench_v2 pipeline):
  {
    "items": [
      {
        "task_id": "...",
        "source": "...",
        "benchmark": "...",
        "expectation": "text|image",
        "input": [
          {"type": "text", "content": "..."},
          {"type": "image", "content": "/path/to/image.jpg"}
        ]
      }
    ]
  }
  or a top-level list of such dicts.

Output per item:
  item_dir/
    input.json, meta.json, output_r00.json, and *.png if generated
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms.functional import to_tensor
from transformers import AutoTokenizer, Qwen2ForCausalLM

from tok.mm_autoencoder import MMAutoEncoder
from tok.ta_tok import TextAlignedTokenizer


# ---------------------------------------------------------------------------
# Default model / tokenizer paths (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "csuhan/Tar-7B-v0.1"
DEFAULT_AR_PATH = None        # auto-download from HF
DEFAULT_ENCODER_PATH = None   # auto-download from HF
DEFAULT_DECODER_PATH = None   # auto-download from HF


def resolve_model_weights(args):
    """Download model weights from HuggingFace if local paths not provided."""
    if args.ar_path is None:
        args.ar_path = hf_hub_download("csuhan/TA-Tok", "ar_dtok_lp_512px.pth")
    if args.encoder_path is None:
        args.encoder_path = hf_hub_download("csuhan/TA-Tok", "ta_tok.pth")
    if args.decoder_path is None:
        args.decoder_path = hf_hub_download(
            "peizesun/llamagen_t2i", "vq_ds16_t2i.pt"
        )


def load_models(args):
    """Load LLM, text tokenizer, visual tokenizer, and image decoder once."""
    device = torch.device(args.device)
    dtype = torch.bfloat16

    # 1) LLM
    print(f"[info] loading LLM from {args.model_path} ...")
    model = Qwen2ForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype
    ).to(device)
    model.eval()

    # 2) Text tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 3) Visual tokenizer (for encoding input images → <I{n}> tokens)
    print(f"[info] loading TA-Tok from {args.encoder_path} ...")
    visual_tokenizer = TextAlignedTokenizer.from_checkpoint(
        args.encoder_path, load_teacher=False, input_type="indices"
    ).to(device)

    # 4) Image decoder (for decoding generated <I{n}> tokens → image)
    print(f"[info] loading MMAutoEncoder ...")
    image_decoder = MMAutoEncoder(
        ar_path=args.ar_path,
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path,
        encoder_args={"input_type": "rec"},
        decoder_args={},
    ).eval().to(dtype=dtype, device=device)
    # cls_token_num and pool_scale are set dynamically per item in parse_output
    return model, tokenizer, visual_tokenizer, image_decoder, device


def encode_image_to_tokens(
    image_path: str,
    visual_tokenizer: TextAlignedTokenizer,
    device: torch.device,
) -> str:
    """Encode an image file into a string of <I{n}> tokens."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = to_tensor(img).unsqueeze(0).to(device)
    codes = visual_tokenizer(img_tensor)["encoded"]  # [1, N]
    token_str = "".join(f"<I{x}>" for x in codes[0].cpu().tolist())
    return token_str


def build_prompt(
    input_segs: List[Dict[str, Any]],
    tokenizer,
    visual_tokenizer: TextAlignedTokenizer,
    device: torch.device,
    inject_boi: bool = False,
    inject_scale: int = 0,
) -> str:
    """
    Build a chat prompt from input segments.

    - text segments → inserted as-is
    - image segments → encoded to <I{n}> tokens via TA-Tok

    If inject_boi=True, appends <im_start><S{scale}> to force image generation.
    Otherwise the model decides the output modality autonomously.
    """
    user_content_parts = []

    for seg in input_segs:
        t = seg["type"]
        c = seg["content"]

        if t == "text":
            user_content_parts.append(c)
        elif t == "image":
            abs_path = os.path.abspath(c)
            img_tokens = encode_image_to_tokens(abs_path, visual_tokenizer, device)
            user_content_parts.append(img_tokens)
        elif t == "sentinel":
            pass
        else:
            raise ValueError(f"Unknown segment type: {t}")

    user_content = "\n".join(user_content_parts)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if inject_boi:
        prompt += f"<im_start><S{inject_scale}>"

    return prompt


# Scale token → (pool_scale, seq_len) mapping
SCALE_MAP = {
    0: (1, 729),   # S0 → 27x27, highest resolution
    1: (2, 169),   # S1 → 13x13
    2: (3, 81),    # S2 → 9x9, lowest resolution
}


def parse_output(
    generated_text: str,
    image_decoder: MMAutoEncoder,
    device: torch.device,
    item_dir: Path,
    repeat_idx: int,
    cfg_scale: float,
) -> List[Dict[str, Any]]:
    """
    Parse the model's generated text into output segments.

    - If <im_start> is found → parse <S{n}> to determine scale/seq_len,
      extract <I{n}> tokens, and decode to image
    - Text before/after image tokens → text segments
    """
    out_segs: List[Dict[str, Any]] = []

    # Check if the model decided to generate an image
    im_start_marker = "<im_start>"
    im_start_pos = generated_text.find(im_start_marker)

    if im_start_pos != -1:
        # Text before image (if any)
        text_before = generated_text[:im_start_pos].strip()
        if text_before:
            out_segs.append({"type": "text", "content": text_before})

        # Extract image part and detect scale
        image_part = generated_text[im_start_pos:]
        scale_match = re.search(r"<S(\d+)>", image_part)
        scale = int(scale_match.group(1)) if scale_match else 0
        pool_scale, seq_len = SCALE_MAP.get(scale, (1, 729))
        print(f"  [scale] S{scale} → pool_scale={pool_scale}, seq_len={seq_len}")

        # Extract image codes
        codes = [int(x) for x in re.findall(r"<I(\d+)>", image_part)]

        if len(codes) > 0:
            # Pad or truncate to detected seq_len
            raw_count = len(codes)
            if len(codes) < seq_len:
                codes = codes + [0] * (seq_len - len(codes))
            else:
                codes = codes[:seq_len]

            code_tensor = torch.tensor(codes).unsqueeze(0).to(device)

            # Dynamically set decoder params to match detected scale
            image_decoder.ar_model.cls_token_num = seq_len
            image_decoder.encoder.pool_scale = pool_scale

            with torch.no_grad():
                gen_tensor = image_decoder.decode_from_encoder_indices(
                    code_tensor, {"cfg_scale": cfg_scale}
                )

            img = Image.fromarray(gen_tensor[0].numpy())
            img_path = item_dir / f"r{repeat_idx:02d}_gen.png"
            img.save(img_path.as_posix())

            out_segs.append({
                "type": "image",
                "path": img_path.as_posix(),
                "scale": scale,
                "num_codes": min(raw_count, seq_len),
            })

            # Text after image codes (if any)
            text_after = re.sub(r"<im_start>|<im_end>|<S\d+>|<I\d+>", "", image_part).strip()
            if text_after:
                out_segs.append({"type": "text", "content": text_after})
        else:
            out_segs.append({
                "type": "image",
                "status": "no_codes",
                "raw": image_part[:200],
            })
    else:
        # Pure text output
        text = generated_text.strip()
        if text:
            out_segs.append({"type": "text", "content": text})

    return out_segs


def compute_im_start_prob(
    model,
    tokenizer,
    input_ids: torch.LongTensor,
    device: torch.device,
) -> Optional[float]:
    """
    Compute the probability of <im_start> at the first generated token position.
    Analogous to Anole's boi_prob.
    """
    im_start_id = tokenizer.convert_tokens_to_ids("<im_start>")
    if im_start_id is None:
        return None

    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
        # logits at the last input position = distribution over first generated token
        logits = outputs.logits[:, -1, :]  # [batch, vocab]
        probs = logits.softmax(dim=-1)
        im_start_prob = probs[0, im_start_id].item()

    return im_start_prob


def run_one(
    model,
    tokenizer,
    visual_tokenizer,
    image_decoder,
    device: torch.device,
    item_dir: Path,
    input_segs: List[Dict[str, Any]],
    args,
    repeat_idx: int = 0,
) -> None:
    """Run inference on a single item."""
    prompt = build_prompt(
        input_segs, tokenizer, visual_tokenizer, device,
        inject_boi=args.inject_boi,
        inject_scale=args.inject_scale,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    # Compute im_start probability (skip if BOI is injected — it's forced)
    im_start_prob = None
    if not args.inject_boi:
        im_start_prob = compute_im_start_prob(model, tokenizer, input_ids, device)
        if im_start_prob is not None:
            print(f"  [im_start_prob] = {im_start_prob:.6f}")
    else:
        print(f"  [inject_boi] scale=S{args.inject_scale}")

    # Max tokens: if BOI injected, we know it's image → use seq_len for the injected scale
    if args.inject_boi:
        _, forced_seq_len = SCALE_MAP.get(args.inject_scale, (1, 729))
        max_new = forced_seq_len + 16
    else:
        max_new = max(args.max_new_tokens, 729 + 16)

    gen_ids = model.generate(
        input_ids,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # Decode only the generated part
    gen_text = tokenizer.decode(
        gen_ids[0, input_ids.shape[1]:], skip_special_tokens=False
    )

    # Strip EOS / padding tokens for cleaner parsing
    for eos_tok in ["<|im_end|>", "<|endoftext|>", "</s>"]:
        gen_text = gen_text.replace(eos_tok, "")

    if args.print_raw:
        print(f"  [raw] {gen_text[:300]}")

    # When BOI is injected, the generated text contains only <I{n}> codes
    # (no <im_start><S{n}> since those were in the prompt, not in generation).
    # Prepend the injected tokens so parse_output can find them.
    if args.inject_boi and "<im_start>" not in gen_text:
        gen_text = f"<im_start><S{args.inject_scale}>" + gen_text

    # Parse output into segments
    out_segs = parse_output(
        generated_text=gen_text,
        image_decoder=image_decoder,
        device=device,
        item_dir=item_dir,
        repeat_idx=repeat_idx,
        cfg_scale=args.cfg_scale,
    )

    # Print results
    for seg in out_segs:
        if seg["type"] == "text" and args.print_text:
            print(f"  [text] {seg['content'][:200]}")
        elif seg["type"] == "image":
            print(f"  [image] {seg.get('path', seg.get('status', '?'))}")

    # Save output JSON
    if args.output_json:
        stem, dot, suf = args.output_json.partition(".")
        name = (
            f"{stem}_r{repeat_idx:02d}{dot}{suf}"
            if dot
            else f"{args.output_json}_r{repeat_idx:02d}"
        )
        out_path = item_dir / name
        output_data = [{"type": "meta", "im_start_prob": im_start_prob}] + out_segs
        out_path.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  [saved] {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Single-load batch runner for Tar-7B multimodal inference. "
        "Model autonomously decides output modality (text vs image)."
    )

    # I/O
    p.add_argument("--input_json", required=True,
                    help="JSON with .items or list of item dicts.")
    p.add_argument("--out_root", required=True,
                    help="Root directory for outputs.")
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--flat", action="store_true",
                    help="Flat output (item_XXXX/) instead of batch/item hierarchy.")

    # Model paths
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH,
                    help="HF model ID or local path for Tar LLM.")
    p.add_argument("--ar_path", default=DEFAULT_AR_PATH,
                    help="AR de-tokenizer checkpoint (auto-download if omitted).")
    p.add_argument("--encoder_path", default=DEFAULT_ENCODER_PATH,
                    help="TA-Tok encoder checkpoint (auto-download if omitted).")
    p.add_argument("--decoder_path", default=DEFAULT_DECODER_PATH,
                    help="VQVAE decoder checkpoint (auto-download if omitted).")
    p.add_argument("--device", default="cuda:0")

    # Image generation params
    p.add_argument("--scale", type=int, default=1,
                    help="Image scale: 0=small(81 tokens), 1=medium(729), 2=large")
    p.add_argument("--seq_len", type=int, default=729,
                    help="Number of image code tokens. 729 for scale=1.")
    p.add_argument("--cfg_scale", type=float, default=4.0,
                    help="Classifier-free guidance scale for image decoding.")

    # Generation params
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=1200)
    p.add_argument("--max_new_tokens", type=int, default=512,
                    help="Max new tokens for text-mode generation.")

    # Output options
    p.add_argument("--output-json", default="output.json")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--print-text", action="store_true", default=True)
    p.add_argument("--print-raw", action="store_true", default=False,
                    help="Print raw generated text (for debugging).")
    p.add_argument("--inject-boi", action="store_true", default=False,
                    help="Force image generation by injecting <im_start><S{scale}> into prompt.")
    p.add_argument("--inject-scale", type=int, default=0,
                    help="Scale to use when --inject-boi is set (0=729, 1=169, 2=81). Default: 0.")

    args = p.parse_args()

    if args.output_json == "":
        args.output_json = None

    # Resolve model weights
    resolve_model_weights(args)

    # Load input data
    input_path = Path(args.input_json)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    if not isinstance(items, list):
        raise ValueError("Expected list under .items or top-level list.")

    total = len(items)
    batch_size = args.batch_size
    batches = math.ceil(total / batch_size) if batch_size > 0 else 0

    print(f"[info] input={input_path} total_items={total} "
          f"batch_size={batch_size} batches={batches}")
    print(f"[info] out_root={out_root}")
    print(f"[info] model={args.model_path} scale={args.scale} seq_len={args.seq_len}")

    # Load models once
    model, tokenizer, visual_tokenizer, image_decoder, device = load_models(args)

    for b in range(batches):
        start = b * batch_size
        end = min(total, (b + 1) * batch_size)
        batch = items[start:end]

        if not args.flat:
            batch_dir = out_root / f"batch_{b:04d}"
            batch_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[batch] {b:04d} items {start}..{end - 1}")

        for k, it in enumerate(batch):
            global_idx = start + k
            if args.flat:
                item_dir = out_root / f"item_{global_idx:04d}"
            else:
                item_dir = batch_dir / f"item_{k:04d}"
            item_dir.mkdir(parents=True, exist_ok=True)

            in_segs = it.get("input")
            if not isinstance(in_segs, list):
                (item_dir / "error.json").write_text(
                    json.dumps(
                        {"error": "missing_input_list", "item": it},
                        ensure_ascii=False, indent=2,
                    ) + "\n",
                    encoding="utf-8",
                )
                continue

            # Save input and meta
            (item_dir / "input.json").write_text(
                json.dumps(in_segs, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            meta = {
                "task_id": it.get("task_id"),
                "source": it.get("source"),
                "benchmark": it.get("benchmark"),
                "expectation": it.get("expectation"),
            }
            (item_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            print(f"  [{global_idx}] task_id={it.get('task_id')} "
                  f"expect={it.get('expectation')}")

            try:
                for r in range(args.repeat):
                    run_one(
                        model=model,
                        tokenizer=tokenizer,
                        visual_tokenizer=visual_tokenizer,
                        image_decoder=image_decoder,
                        device=device,
                        item_dir=item_dir,
                        input_segs=in_segs,
                        args=args,
                        repeat_idx=r,
                    )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                (item_dir / "error.json").write_text(
                    json.dumps({"error": str(e), "traceback": tb}, ensure_ascii=False, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
                print(f"  [error] {e}\n{tb}")

    print("\n[done] Tar single-load run complete")


if __name__ == "__main__":
    main()
