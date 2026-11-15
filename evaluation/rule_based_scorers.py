#!/usr/bin/env python3
"""Rule-based scorers for the four COMI-Bench sources that don't need a judge
model: gsm8k, VQAv2, mmlu, ScienceQA. Each takes (gold, response_text) and
returns a 0-100 score. Attach the result to each item's `score` field before
running compute_msa_tp.py.

mmlu and ScienceQA are open-ended (no multiple-choice letters in the
prompt -- see pipeline/stage0/scienceqa_openstyle/ for why), so there is no
exact-match target. `score_open_style` uses loose containment as a cheap
proxy; it under-counts paraphrased correct answers, which is why the paper's
mmlu/ScienceQA numbers come from an LLM judge instead (see evaluation/README.md)
-- these rule-based versions are provided for quick local sanity-checking,
not as the source of the paper's reported numbers for those two sources.
"""
import re

ARTICLES = {"a", "an", "the"}
NUMBER_WORDS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_PUNCT_RE = re.compile(r"[^\w\s']")
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _vqa_normalize(s: str) -> str:
    """Official VQA accuracy normalization: lowercase, strip punctuation,
    spell out number words, drop articles."""
    s = s.lower().strip().replace("-", " ").replace("/", " ")
    s = _PUNCT_RE.sub("", s)
    words = [NUMBER_WORDS.get(w, w) for w in s.split()]
    words = [w for w in words if w not in ARTICLES]
    return " ".join(words)


def score_vqa(gold: str, response_text: str) -> float:
    g, p = _vqa_normalize(gold), _vqa_normalize(response_text)
    if not g:
        return 0.0
    hit = g in p.split() or f" {g} " in f" {p} " or p == g
    return 100.0 if hit else 0.0


def score_open_style(gold: str, response_text: str) -> float:
    """Loose containment, for mmlu / ScienceQA (open-ended, no MC letters)."""
    g, p = _vqa_normalize(gold), _vqa_normalize(response_text)
    if not g:
        return 0.0
    return 100.0 if g in p else 0.0


def score_gsm8k(gold: str, response_text: str) -> float:
    """Last number in the response, trailing period stripped, compared to
    gold (commas stripped from both). Matching ONLY the last number -- not
    "any number that appears anywhere in the response" -- matters: GSM8K
    responses often restate intermediate quantities from the problem itself,
    and an any-number match scores those as correct even when the model's
    final stated answer is wrong."""
    gold_c = gold.replace(",", "").strip()
    nums = [n.replace(",", "") for n in _NUMBER_RE.findall(response_text)]
    if not nums:
        return 0.0
    return 100.0 if nums[-1].rstrip(".") == gold_c else 0.0
