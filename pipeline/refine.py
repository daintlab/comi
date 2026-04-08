#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refine script (generate N diverse refinements per sample)

- Reads JSON with {"results":[...]} items
- Generates N refined prompt variants per sample
  * default: only for expected != "expected" (not_expected/unknown)
  * optional: include expected samples too (--include_expected)
- Preserves original meaning, but makes the intended modality instruction more explicit
- Strong diversity constraints to avoid template collapse (esp. t2i)
- Writes output JSON with a new field:
    refined_variants: [v1, v2, ... vN]

API key loading:
- Loads OPENAI_API_KEY from .env (python-dotenv), then falls back to environment variable.

Usage:
  python refine.py \
    --input_json /path/in.json \
    --output_json /path/out.json \
    --model gpt-5.2 \
    --n_variants 5 \
    --include_expected \
    --skip_existing \
    --save_every 25 \
    --record_failures

Requirements:
  pip install openai python-dotenv tqdm
"""

import argparse
import json
import os
import random
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import dotenv
from openai import OpenAI

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

REFINER_SYSTEM_PROMPT = r"""
You are a careful prompt refiner.

You will be given ONE sample:
- task_id, source, original_question
- expected, expected_reason, refine_comment

Your job:
Generate N diverse refined prompt variants under two core principles.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRINCIPLE 1: SEMANTIC UNIT PRESERVATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every variant must fully preserve the semantic unit of the original prompt.

The semantic unit is determined by content length and completeness:

[Complete, self-contained content — gsm8k, mmlu, dpg_bench, ScienceQA, VQAv2]
  The entire original text is the semantic unit.
  → Preserve it VERBATIM in every variant. Do NOT summarize, shorten, or paraphrase.
  → Only ADD instruction framing before or after the original text.
  → Structure: [instruction framing] + [original verbatim]
               OR [original verbatim] + [instruction framing]

[Short label or brief instruction — geneval, MagicBrush, ImgEditBench]
  Key semantic attributes (objects, colors, counts, spatial relations, actions) are the semantic unit.
  → You MAY paraphrase the surface form to increase structural diversity.
  → All key attributes must be preserved. No attributes may be added or removed.
  → Do NOT append the original verbatim after refined framing if the meaning is already incorporated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRINCIPLE 2: NON-REDUNDANCY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Added instruction framing must not duplicate meaning already present in the original.
- Framing should only ADD what is missing: modality cue, output format, or image reference.
- Do not re-state constraints already present in the original prompt.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODALITY GUIDANCE BY SOURCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- gsm8k, mmlu              => text answer (t2t)
- geneval, dpg_bench       => image generation (t2i) [ADD explicit generation verb]
- VQAv2, ScienceQA         => image-grounded text answer (ti2t) [ADD explicit image reference]
- MagicBrush, ImgEditBench => image editing (ti2i) [ADD explicit edit + image reference]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIVERSITY REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- At least 4/5 variants must have different first 3-6 words.
- Do NOT repeat the same leading phrase across variants.
- Use different directive styles (no style repeated more than twice):
  (1) direct imperative        — "Generate / Edit / Solve..."
  (2) polite request           — "Please create / modify..."
  (3) task framing             — "Your task is to..."
  (4) output-spec-first        — "Output format: an image / a number..."
  (5) question form            — "Can you...?" / "Could you...?"
  (6) constraint-first         — "Only change X; keep everything else..."
  (7) step-lite two-sentence   — "First look at the image. Then answer..."
  (8) light role framing       — "Act as an image editor..."
  (9) minimal strict form      — "Return only the letter..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Do NOT answer the question.
- Do NOT add new constraints not present in the original.
- Do NOT add safety/ethics disclaimers.
- End every variant with a period or question mark.
- Output MUST be strict JSON: {"refined_variants": ["...", "...", ...]}
- Return exactly N variants.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEHAVIOR BY EXPECTED STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If expected="expected":
  Produce cleaner or more explicit variants; keep changes minimal.
  Focus on output format clarity.

If expected="not_expected":
  Follow refine_comment closely to make the intended modality explicit.
  Ensure every variant satisfies the modality criteria for its source.
""".strip()



CODE_FENCE_START_RE = re.compile(r"^\s*```(?:json)?\s*\n", re.IGNORECASE)
CODE_FENCE_END_RE = re.compile(r"\n```[\s]*$", re.IGNORECASE)


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = CODE_FENCE_START_RE.sub("", s)
    s = CODE_FENCE_END_RE.sub("", s)
    return s.strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Try json.loads; else extract first balanced {...} and json.loads it."""
    text = _strip_code_fences(text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("no_json_object_found")

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
                break

    raise ValueError("json_parse_failed")


def _is_retryable_error(msg: str) -> bool:
    s = (msg or "").lower()
    retry_markers = [
        "rate limit", "429",
        "timeout",
        "temporarily",
        "overloaded",
        "server error", "500", "502", "503", "504",
        "connection",
        "unavailable",
        "try again",
    ]
    return any(m in s for m in retry_markers)


def _sleep_backoff(attempt: int, base: float = 0.8, cap: float = 15.0) -> None:
    delay = min(cap, base * (2 ** attempt))
    jitter = random.uniform(0.0, 0.2 * delay)
    time.sleep(delay + jitter)


def _dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        x2 = (x or "").strip()
        if not x2:
            continue
        if x2 in seen:
            continue
        seen.add(x2)
        out.append(x2)
    return out


def _load_api_key() -> str:
    # Load .env from current working directory (and parents) if present
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Put it in .env or export it in the environment.")
    return api_key


def generate_refinements(
    client: OpenAI,
    model: str,
    item: Dict[str, Any],
    n_variants: int = 5,
    temperature: float = 0.7,
    max_retries: int = 4,
) -> List[str]:
    payload = {
        "task_id": item.get("task_id"),
        "source": item.get("source"),
        "original_question": item.get("original_question"),
        "expected": item.get("expected"),
        "expected_reason": item.get("expected_reason", ""),
        "refine_comment": item.get("refine_comment", ""),
        "n_variants": n_variants,
    }

    user_prompt = (
        f"Generate exactly {n_variants} refined prompt variants.\n"
        f"Return JSON: {{\"refined_variants\":[...]}}, with length exactly {n_variants}.\n"
        f"Sample:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    messages = [
        {"role": "system", "content": REFINER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            # Prefer strict JSON output if available
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                text = resp.choices[0].message.content or ""
            except TypeError:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                text = resp.choices[0].message.content or ""

            obj = _extract_json_object(text)
            variants = obj.get("refined_variants", [])

            if not isinstance(variants, list):
                raise ValueError("refined_variants_not_list")

            variants = [str(v).replace("\n", " ").replace("\r", " ").strip() for v in variants]
            variants = _dedupe_keep_order(variants)

            # Soft enforcement: if too few, retry (or return whatever)
            if len(variants) >= n_variants:
                return variants[:n_variants]

            last_err = f"too_few_variants:{len(variants)}"
            if attempt < max_retries:
                _sleep_backoff(attempt, base=0.6, cap=10.0)
                continue
            return variants

        except Exception as e:
            last_err = str(e)
            if attempt >= max_retries:
                break
            if _is_retryable_error(last_err) or "json_" in last_err or "parse" in last_err or "too_few_variants" in (last_err or ""):
                _sleep_backoff(attempt)
                continue
            break

    raise RuntimeError(f"refine_failed: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="Path to input JSON (dict with results[])")
    ap.add_argument("--output_json", required=True, help="Output JSON path")
    ap.add_argument("--model", default="gpt-5.2", help="Model name")
    ap.add_argument("--n_variants", type=int, default=5, help="How many refined variants per sample")
    ap.add_argument("--include_expected", action="store_true", help="Also generate variants for items where expected == 'expected'")
    ap.add_argument("--max_items", type=int, default=0, help="0 means all eligible")
    ap.add_argument("--sleep_sec", type=float, default=0.05, help="Base sleep between items")
    ap.add_argument("--save_every", type=int, default=25, help="Periodic checkpoint writes")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if refined_variants already exist with >= n_variants")
    ap.add_argument("--record_failures", action="store_true", help="On failure, store _refine_error fields and continue")
    args = ap.parse_args()

    api_key = _load_api_key()
    client = OpenAI(api_key=api_key)

    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "results" not in data or not isinstance(data["results"], list):
        raise ValueError("input_json must be a dict with results[]")

    out = deepcopy(data)
    results = out["results"]

    eligible_indices: List[int] = []
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            continue

        # Default: refine only not_expected/unknown; optional: include expected too
        if (not args.include_expected) and r.get("expected") == "expected":
            continue

        if args.skip_existing and isinstance(r.get("refined_variants"), list) and len(r["refined_variants"]) >= args.n_variants:
            continue

        eligible_indices.append(i)

    iterator = eligible_indices
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(eligible_indices), desc="refine_prompts", unit="item")

    processed = 0
    for idx in iterator:
        r = results[idx]
        try:
            variants = generate_refinements(
                client=client,
                model=args.model,
                item=r,
                n_variants=args.n_variants,
            )
            r["refined_variants"] = variants
            r.pop("_refine_error", None)
            r.pop("_refine_error_at", None)

        except Exception as e:
            if not args.record_failures:
                raise
            r["refined_variants"] = r.get("refined_variants", [])
            r["_refine_error"] = str(e)
            r["_refine_error_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        processed += 1
        if args.max_items and processed >= args.max_items:
            break

        if args.save_every and processed % args.save_every == 0:
            Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()