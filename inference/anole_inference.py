#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-process batch runner for multimodal_inference.py inputs.

Why:
- run_merged_all_in_batches.sh spawns a new Python process per item, reloading the model every time.
- This script loads the model ONCE and iterates over items, dramatically reducing overhead.

Input:
- JSON produced by convert_merged_all_to_inputs.py (or compatible):
  - either {"items":[{task_id, source, benchmark, expectation, input:[{type,content},...]}, ...]}
  - or a top-level list of such item dicts

Output:
- OUT_ROOT/batch_0000/item_0000/ ...
  - input.json, meta.json, output.json, and generated *.png if any
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

from chameleon.inference.chameleon import ChameleonInferenceModel, Options
from constants import (
    MODEL_7B_PATH,
    TOKENIZER_TEXT_PATH,
    TOKENIZER_IMAGE_CFG_PATH,
    TOKENIZER_IMAGE_PATH,
)


def split_interleaved(
    tokens: torch.LongTensor,
    boi: int,
    eoi: int,
) -> List[Tuple[str, List[int]]]:
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError(f"Expected tokens shape (1, L), got {tuple(tokens.shape)}")

    ids: List[int] = tokens[0].tolist()
    segs: List[Tuple[str, List[int]]] = []
    cur: List[int] = []
    in_image = False

    for tok in ids:
        if tok == boi:
            if cur:
                segs.append(("text", cur))
                cur = []
            in_image = True
            continue

        if tok == eoi and in_image:
            segs.append(("image", cur))
            cur = []
            in_image = False
            continue

        cur.append(tok)

    if cur:
        segs.append(("image" if in_image else "text", cur))

    return segs


def build_prompt_ui(
    input_segs: List[Dict[str, Any]],
    no_eot: bool,
) -> List[List[Dict[str, Any]]]:
    batch_prompt_ui: List[List[Dict[str, Any]]] = [[]]

    for seg in input_segs:
        t = seg["type"]
        c = seg["content"]

        if t == "text":
            batch_prompt_ui[0].append({"type": "text", "value": c})
        elif t == "image":
            abs_path = os.path.abspath(c)
            batch_prompt_ui[0].append({"type": "image", "value": f"file:{abs_path}"})
        elif t == "sentinel":
            batch_prompt_ui[0].append({"type": "sentinel", "value": c})
        else:
            raise ValueError(f"Unknown segment type: {t}")

    if not no_eot:
        batch_prompt_ui[0].append(
            {"type": "sentinel", "value": "<END-OF-TURN>"}
        )

    return batch_prompt_ui


def run_one(
    model: ChameleonInferenceModel,
    options: Options,
    item_dir: Path,
    input_segs: List[Dict[str, Any]],
    no_eot: bool,
    output_json_name: str,
    print_text: bool,
    print_images: bool,
    repeat_idx: int = 0,
) -> None:
    batch_prompt_ui = build_prompt_ui(input_segs, no_eot=no_eot)

    # Stream tokens to capture first generated token's logits for BOI prob
    first_gen_logits = None
    token_ids = []
    for tok in model.stream(batch_prompt_ui=batch_prompt_ui, options=options):
        token_ids.append(tok.id)
        if first_gen_logits is None and tok.logits is not None:
            first_gen_logits = tok.logits  # [batch, vocab]

    if not token_ids:
        tokens = torch.LongTensor()
    else:
        tokens = torch.stack(token_ids).T

    # Compute BOI probability from first generated token
    boi, eoi = model.vocab.begin_image, model.vocab.end_image
    print(f"boi: {boi} ----- eoi: {eoi}") 
    boi_prob = None
    if first_gen_logits is not None:
        probs = first_gen_logits.softmax(dim=1)  # [batch, vocab]
        boi_prob = probs[0, boi].item()
        print(f"  [boi_prob] first token BOI prob = {boi_prob:.6f}")

    segs = split_interleaved(tokens, boi, eoi)

    item_dir.mkdir(parents=True, exist_ok=True)
    out_segs: List[Dict[str, Any]] = []

    device = "cuda"
    for idx, (seg_type, seg_ids) in enumerate(segs):
        if seg_type == "text":
            text = model.decode_text([seg_ids])[0]
            if print_text:
                print(text, end="" if text.endswith("\n") else "\n")
            out_segs.append({"type": "text", "content": text})
        else:
            if len(seg_ids) != 1024:
                print(
                    f"[warn] partial image segment: len={len(seg_ids)} (expected 1024). "
                    f"skipping decode. (segment idx={idx})"
                )
                out_segs.append(
                    {
                        "type": "image",
                        "status": "partial",
                        "token_len": len(seg_ids),
                    }
                )
                continue

            img_tokens = torch.tensor(
                seg_ids,
                dtype=torch.long,
                device=device,
            ).reshape(1, -1)

            img: Image.Image = model.decode_image(img_tokens)[0]
            image_path = item_dir / f"r{repeat_idx:02d}_{idx}.png"
            img.save(image_path.as_posix())

            if print_images:
                print(f"<img: {image_path}>")

            out_segs.append({"type": "image", "path": image_path.as_posix()})

    if output_json_name:
        stem, dot, suf = output_json_name.partition(".")
        name = (
            f"{stem}_r{repeat_idx:02d}{dot}{suf}"
            if dot
            else f"{output_json_name}_r{repeat_idx:02d}"
        )
        out_path = item_dir / name
        output_data = [{"type": "meta", "boi_prob": boi_prob}] + out_segs
        out_path.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[saved] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Single-load batch runner for Chameleon multimodal inference."
    )
    p.add_argument(
        "--input_json",
        required=True,
        help="JSON with .items or list of items containing 'input'.",
    )
    p.add_argument(
        "--out_root",
        required=True,
        help="Root directory to write batch/item outputs.",
    )
    p.add_argument("--batch_size", type=int, default=10)

    p.add_argument("--text-only", action="store_true", help="Generate text only.")
    p.add_argument("--image-only", action="store_true", help="Generate image only.")
    p.add_argument(
        "--no-eot",
        action="store_true",
        help="Do not append <END-OF-TURN> automatically.",
    )

    p.add_argument(
        "--output-json",
        default="output.json",
        help="Output JSON filename per item (empty to disable).",
    )
    p.add_argument("--print-text", action="store_true", default=True)
    p.add_argument("--print-images", action="store_true", default=True)
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat generation per item N times (saved as *_rXX.json).",
    )
    p.add_argument(
        "--flat",
        action="store_true",
        help="Save items flat (item_XXXX/) instead of batch_XXXX/item_XXXX/.",
    )

    args = p.parse_args()

    if args.output_json == "":
        args.output_json = None

    if args.text_only and args.image_only:
        raise ValueError("Cannot set both --text-only and --image-only.")

    input_path = Path(args.input_json)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) and "items" in data else data

    if not isinstance(items, list):
        raise ValueError("Expected list under .items or top-level list.")

    total = len(items)
    batch_size = int(args.batch_size)
    batches = math.ceil(total / batch_size) if batch_size > 0 else 0

    print(
        f"[info] input={input_path} total_items={total} "
        f"batch_size={batch_size} batches={batches}"
    )
    print(f"[info] out_root={out_root}")

    # Load model once
    model = ChameleonInferenceModel(
        MODEL_7B_PATH.as_posix(),
        TOKENIZER_TEXT_PATH.as_posix(),
        TOKENIZER_IMAGE_CFG_PATH.as_posix(),
        TOKENIZER_IMAGE_PATH.as_posix(),
    )

    options = Options()
    if args.text_only:
        options.img = False
    if args.image_only:
        options.txt = False

    for b in range(batches):
        start = b * batch_size
        end = min(total, (b + 1) * batch_size)
        batch = items[start:end]

        if not args.flat:
            batch_dir = out_root / f"batch_{b:04d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
        print(f"[batch] {b:04d} items {start}..{end - 1}")

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
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                continue

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

            try:
                rep = int(args.repeat)
                if rep < 1:
                    raise ValueError("--repeat must be >= 1")

                for r in range(rep):
                    run_one(
                        model=model,
                        options=options,
                        item_dir=item_dir,
                        input_segs=in_segs,
                        no_eot=args.no_eot,
                        output_json_name=args.output_json or "",
                        print_text=args.print_text,
                        print_images=args.print_images,
                        repeat_idx=r,
                    )
            except Exception as e:
                (item_dir / "error.json").write_text(
                    json.dumps({"error": str(e)}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"[error] batch={b:04d} item={k:04d} err={e}")

    print("[done] single-load run complete")


if __name__ == "__main__":
    main()