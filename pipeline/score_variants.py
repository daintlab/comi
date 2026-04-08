#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute & store embedding cosine similarities for refined_variants (no filtering by default).

Why:
- You can tune thresholds later WITHOUT re-embedding/re-computing.
- Optionally, you can also write kept/dropped lists for a chosen threshold at the same time.

Input:
- JSON with {"results":[...]} items containing:
  - original_question (str)
  - refined_variants (list[str])

Output (adds fields per item):
- refined_variants_scores: [
    {"variant": "...", "cosine": 0.812345, "kept": true/false (optional)}
  ]
- (optional) refined_variants_kept / refined_variants_dropped if --threshold is provided

API key:
- Loads OPENAI_API_KEY from .env (python-dotenv), else environment.

Usage:
  pip install openai python-dotenv tqdm numpy
  python score_variants.py \
    --input_json /path/refined.json \
    --output_json /path/refined.scored.json \
    --emb_model text-embedding-3-small

Optional thresholding (still no re-embed; uses stored cosine this run):
  python score_variants.py ... --threshold 0.70 --keep_top_k 0 --min_keep 0

Later, if you already have refined_variants_scores saved,
you can re-run with --reuse_scores to ONLY apply new thresholding without embedding calls:
  python score_variants.py \
    --input_json /path/refined.scored.json \
    --output_json /path/refined.scored.tuned.json \
    --reuse_scores \
    --threshold 0.75
"""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dotenv
import numpy as np
from openai import OpenAI

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

SOURCES = {
    "gsm8k", "mmlu",
    # ti2t: 이미지 참조 추가가 refinement 목적 → 많이 달라질 수 있음
    "VQAv2", "ScienceQA",
    # t2i: 생성 동사 추가가 refinement 목적 → 많이 달라질 수 있음
    "geneval", "dpg_bench",
    # ti2i: 편집 지시 추가가 refinement 목적 → 많이 달라질 수 있음
    "MagicBrush", "ImgEditBench",
}

def get_threshold(r: Dict[str, Any], default_threshold: float) -> float:
    """
    not_expected 샘플은 refinement으로 원본과 많이 달라지는 게 정상이므로
    threshold를 낮게 적용한다.
    expected 샘플은 원본이 이미 좋으므로 기본 threshold를 유지한다.
    """
    expected = r.get("expected", "expected")
    source = r.get("source", "")

    if expected == "not_expected" and source in SOURCES:
        return 0.45
    return default_threshold

def load_api_key() -> str:
    dotenv.load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not found in .env or environment.")
    return key


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed_texts(client: OpenAI, model: str, texts: List[str], batch_size: int = 128) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=chunk)
        for d in resp.data:
            out.append(np.array(d.embedding, dtype=np.float32))
    return out


def apply_thresholding(
    r: Dict[str, Any],
    threshold: float,
    keep_top_k: int = 0,
    min_keep: int = 0,
) -> None:
    """
    Uses r["refined_variants_scores"] to create kept/dropped lists and set kept flags.
    Does NOT compute embeddings.
    """
    scores = r.get("refined_variants_scores")
    if not isinstance(scores, list) or not scores:
        return

    scored: List[Tuple[str, float]] = []
    for row in scores:
        if not isinstance(row, dict):
            continue
        v = row.get("variant")
        s = row.get("cosine")
        if isinstance(v, str) and isinstance(s, (int, float)):
            scored.append((v, float(s)))

    if not scored:
        return

    kept = [(v, s) for v, s in scored if s >= threshold]
    kept_sorted = sorted(kept, key=lambda x: x[1], reverse=True)

    if keep_top_k and keep_top_k > 0:
        kept_sorted = kept_sorted[:keep_top_k]

    if min_keep and min_keep > 0 and len(kept_sorted) < min_keep:
        all_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        kept_sorted = all_sorted[:min_keep]

    kept_set = set(v for v, _ in kept_sorted)

    # update flags
    new_scores = []
    for v, s in scored:
        new_scores.append({"variant": v, "cosine": round(s, 6), "kept": v in kept_set})
    r["refined_variants_scores"] = new_scores

    r["refined_variants_kept"] = [v for v, _ in kept_sorted]
    r["refined_variants_dropped"] = [v for v, _ in scored if v not in kept_set]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--output_json", required=True)

    ap.add_argument("--emb_model", default="text-embedding-3-small")
    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument(
        "--compare_field",
        default="original_question",
        choices=["original_question", "refined_prompt", "interrogative", "declarative"],
        help="Which field to compare variants against for similarity scoring.",
    )

    ap.add_argument(
        "--reuse_scores",
        action="store_true",
        help="Do NOT call embeddings. Only apply thresholding using existing refined_variants_scores.",
    )

    ap.add_argument("--threshold", type=float, default=None, help="If set, also compute kept/dropped using this threshold")
    ap.add_argument("--keep_top_k", type=int, default=0)
    ap.add_argument("--min_keep", type=int, default=0)

    args = ap.parse_args()

    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "results" not in data or not isinstance(data["results"], list):
        raise ValueError("input_json must be a dict with results[]")

    out = deepcopy(data)
    results: List[Dict[str, Any]] = out["results"]

    # If we reuse, just threshold and write out
    if args.reuse_scores:
        if args.threshold is None:
            # still write out unchanged
            Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return

        iterator = results
        if tqdm is not None:
            iterator = tqdm(iterator, desc="apply-threshold", unit="item")

        for r in iterator:
            if not isinstance(r, dict):
                continue
            effective_threshold = get_threshold(r, args.threshold)
            apply_thresholding(r, effective_threshold, args.keep_top_k, args.min_keep)

        Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    # Otherwise compute embeddings & store cosine
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    iterator = results
    if tqdm is not None:
        iterator = tqdm(iterator, desc="score-variants", unit="item")

    for r in iterator:
        if not isinstance(r, dict):
            continue

        base = r.get(args.compare_field)
        variants = r.get("refined_variants")

        if not isinstance(base, str) or not base.strip():
            continue
        if not isinstance(variants, list) or not variants:
            continue

        base = base.strip()
        vlist = [str(v).strip() for v in variants if str(v).strip()]
        if not vlist:
            continue

        texts = [base] + vlist
        embs = embed_texts(client, args.emb_model, texts, batch_size=args.batch_size)
        emb_base = embs[0]
        emb_vars = embs[1:]

        scored_rows = []
        for v, e in zip(vlist, emb_vars):
            s = cosine_sim(emb_base, e)
            scored_rows.append({"variant": v, "cosine": round(s, 6)})

        r["refined_variants_scores"] = scored_rows

        # optional thresholding in the same run
        if args.threshold is not None:
            effective_threshold = get_threshold(r, args.threshold)
            apply_thresholding(r, effective_threshold, args.keep_top_k, args.min_keep)

    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()