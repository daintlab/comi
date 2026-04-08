#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Instruction-completeness gate (minimal, NO strict comment parsing)

Goal:
- Given each sample's original_question, decide whether the prompt TEXT ALONE
  is COMPLETE and EXPLICIT enough to elicit the intended modality for its dataset source.
- Add three keys per item:
  - expected: one of ["expected", "not_expected", "unknown"]
  - expected_reason: one-line reason why it is expected / not_expected / unknown
  - refine_comment: one-line suggestion to make it more reliably elicit the intended modality
    * If expected="expected": suggest an improvement (optional but helpful)
    * If expected!="expected": suggest how to refine it to become expected

Notes:
- We intentionally do NOT validate formats. Only minimal sanity + newline stripping.
- Uses response_format json_object when supported.

Usage:
  OPENAI_API_KEY=... python expected_gate_v2.py \
    --input_json merged.json \
    --output_json expected.json \
    --model gpt-5.2 \
    --skip_existing \
    --save_every 50

Tip:
  pip install tqdm
"""

import argparse
import json
import os
import random
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


SYSTEM_PROMPT = r"""
You are a strict gate that checks whether a prompt is COMPLETE and EXPLICIT enough
to serve as a downstream instruction for the output modality implied by its `source`.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{"task_id": "...", "source": "...", "original_question": "..."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT  (strict JSON, no extra keys, all values single-line)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "expected": "expected|not_expected|unknown",
  "expected_reason": "<one line>",
  "refine_comment": "<one line>"
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE → INTENDED MODALITY MAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
t2t  (text in  → text out)  : gsm8k, mmlu
t2i  (text in  → image out) : geneval, dpg_bench
ti2t (image+text in → text out) : VQAv2, ScienceQA
ti2i (image+text in → image out): MagicBrush, ImgEditBench

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LABEL DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

expected
  The prompt text alone EXPLICITLY and UNAMBIGUOUSLY requests the intended output modality.
  Per-source criteria:

  [t2t — gsm8k, mmlu]
    PASS if the prompt clearly requests a text answer:
      question form, "solve", "explain", "choose", "calculate", etc.
    FAIL if the prompt is only a bare statement with no instruction.

  [t2i — geneval, dpg_bench]
    PASS only if an EXPLICIT generation verb is present:
      "generate", "create", "draw", "render", "make", "produce" + image/photo/picture/...
    FAIL (not_expected) if the prompt is a bare descriptive caption with no generation verb.
      ✗ "a photo of a green tennis racket and a black dog"
      ✓ "Generate a photo of a green tennis racket and a black dog."
    NOTE: "a photo of ..." without a generation verb is NOT sufficient — it is still not_expected.

  [ti2t — VQAv2, ScienceQA]
    PASS only if the prompt EXPLICITLY references a visual input:
      "in the image/photo/picture/diagram/chart", "based on the image",
      "what is shown", "look at the image", "from the figure", etc.
    FAIL if the question is answerable from text alone without any image reference.
    IMPORTANT for ScienceQA: the `hint` field is intentionally excluded from our prompts.
      Do NOT assume hint text is present. Judge ONLY from `original_question`.
      A ScienceQA question that is answerable from hint text alone (without the image)
      must be marked not_expected.

  [ti2i — MagicBrush, ImgEditBench]
    PASS if the prompt explicitly requests editing a given image:
      contains edit language such as "edit", "change", "add", "remove", "replace",
      "make [X] do/be [Y]", "turn [X] into [Y]", etc.
    FAIL if the prompt describes a scene or action without referencing an existing image to edit.
      ✗ "He should be eating a watermelon"  (no explicit edit instruction)
      ✓ "Edit the image so that he is eating a watermelon."

not_expected
  The prompt is incomplete or underspecified for the intended modality.
  Assign this whenever the criteria above are NOT met.

unknown
  Use ONLY when the prompt is genuinely ambiguous and neither expected nor not_expected
  can be assigned with reasonable confidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

expected_reason (one line):
  - State WHY the label was assigned.
  - Quote 1–2 short cues directly from the prompt using double quotes.
  - Do NOT rewrite or paraphrase the full question.
  - Do NOT answer the question.

refine_comment (one line):
  - If expected="expected": suggest a small, specific improvement to make the modality
    even more unambiguous (e.g., add "based on the image", add "output only the letter").
  - If expected!="expected": give a concrete one-line rewrite suggestion that would
    make the prompt clearly expected for the intended modality.
  - Must be actionable. Must be one line.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Judge ONLY from `original_question`. Do NOT assume hint, choices, or image are present.
- Do NOT output extra JSON keys.
- Do NOT use newlines inside any field value.
- Do NOT answer the question being evaluated.
""".strip()

ALLOWED_EXPECTED = {"expected", "not_expected", "unknown"}

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


def _sleep_backoff(attempt: int, base: float = 0.7, cap: float = 12.0) -> None:
    delay = min(cap, base * (2 ** attempt))
    jitter = random.uniform(0.0, 0.2 * delay)
    time.sleep(delay + jitter)


def classify_expected(
    client: OpenAI,
    model: str,
    task_id: Any,
    source: Any,
    original_question: Any,
    temperature: float = 0.0,
    max_retries: int = 4,
) -> Dict[str, str]:
    payload = {"task_id": task_id, "source": source, "original_question": original_question}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
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

            expected = obj.get("expected", "unknown")
            expected_reason = obj.get("expected_reason", "")
            refine_comment = obj.get("refine_comment", "")

            # minimal sanity
            if expected not in ALLOWED_EXPECTED:
                expected = "unknown"

            if not isinstance(expected_reason, str):
                expected_reason = str(expected_reason)
            if not isinstance(refine_comment, str):
                refine_comment = str(refine_comment)

            expected_reason = expected_reason.replace("\n", " ").replace("\r", " ").strip()
            refine_comment = refine_comment.replace("\n", " ").replace("\r", " ").strip()

            # guard against common mistake: model echoes dataset name in expected
            if isinstance(expected, str) and expected.lower() in {
                "gsm8k", "mmlu", "vqav2", "scienceqa", "infovqa", "editbench", "imgeditbench", "magicbrush", "flickr30k", "geneval", "dpg_bench", "humaneval"
            }:
                expected = "unknown"

            return {
                "expected": expected,
                "expected_reason": expected_reason,
                "refine_comment": refine_comment,
            }

        except Exception as e:
            last_err = str(e)
            if attempt >= max_retries:
                break
            if _is_retryable_error(last_err) or "json_" in last_err or "parse" in last_err:
                _sleep_backoff(attempt)
                continue
            break

    raise RuntimeError(f"classify_failed: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="Path to input JSON (dict with results[])")
    ap.add_argument("--output_json", required=True, help="Output JSON path")
    ap.add_argument("--model", default="gpt-5.2", help="Model name")
    ap.add_argument("--max_items", type=int, default=0, help="0 means all")
    ap.add_argument("--sleep_sec", type=float, default=0.05, help="Base sleep between items")
    ap.add_argument("--save_every", type=int, default=50, help="Periodic checkpoint writes")
    ap.add_argument("--skip_existing", action="store_true", help="Skip items already have expected+expected_reason+refine_comment")
    ap.add_argument("--record_failures", action="store_true", help="On failure, write unknown + _error fields")
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment.")

    inp = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    if not isinstance(inp, dict) or "results" not in inp or not isinstance(inp["results"], list):
        raise ValueError("input_json must be a dict with results[]")

    out = deepcopy(inp)
    client = OpenAI(api_key=api_key)

    results = out["results"]
    iterator = range(len(results))
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(results), desc="expected_gate_v2", unit="item")

    processed = 0
    for idx in iterator:
        r = results[idx]
        if not isinstance(r, dict):
            continue

        if args.skip_existing and all(k in r for k in ("expected", "expected_reason", "refine_comment")):
            continue

        try:
            ann = classify_expected(
                client=client,
                model=args.model,
                task_id=r.get("task_id"),
                source=r.get("source"),
                original_question=r.get("original_question"),
            )
            r["expected"] = ann["expected"]
            r["expected_reason"] = ann["expected_reason"]
            r["refine_comment"] = ann["refine_comment"]
            r.pop("_error", None)
            r.pop("_error_at", None)

        except Exception as e:
            if not args.record_failures:
                raise
            r["expected"] = "unknown"
            r["expected_reason"] = "classification_failed"
            r["refine_comment"] = "classification_failed"
            r["_error"] = str(e)
            r["_error_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

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