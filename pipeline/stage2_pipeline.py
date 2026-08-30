"""
Stage 2a (수정) -> 2b (증강) -> 3 (검증).

Stage 1은 stage1_judge.py로 미리 돌려두고, 그 출력 JSON을 입력으로 받는다.
Stage 1을 두 번 돌리지 않기 위해 분리했다.

  입력    stage1_judge.py의 출력 (results[]에 route, needs_refinement 포함)
  2a      route == "fix" 인 샘플만 수정. keep은 원본 그대로 통과.
  2b      전체 샘플에 대해 의문형 2 / 비의문형 3으로 variant 5개
  3       variant별 modality / meaning / naturalness 3축 검증

Stage 2a는 refine_hint를 쓰지 않는다. 재작성 원칙을 시스템 프롬프트에
고정해 두었다. hint를 쓰면 judge의 판정 근거가 그대로 처방이 되어
모든 프롬프트에 "Generate an image:"가 붙는다.

사용법
------
    python stage2_pipeline.py --input stage1_55.json --output stage2_55.json
    python stage2_pipeline.py --input stage1_55.json --limit 10 --workers 4
    GEN_MODEL=gpt-5.5 VAL_MODEL=gpt-5.5 python stage2_pipeline.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GEN_MODEL = os.getenv("GEN_MODEL", "gpt-5.5")
VAL_MODEL = os.getenv("VAL_MODEL", "gpt-5.5")

# 시스템 프롬프트는 prompts/ 아래 txt로 관리한다. 스크립트를 건드리지 않고
# 프롬프트만 수정할 수 있게 하기 위함.
PROMPT_DIR = Path(os.getenv("PROMPT_DIR", "prompts"))


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    if not path.exists():
        print(f"프롬프트 파일이 없습니다: {path}", file=sys.stderr)
        raise SystemExit(1)
    return path.read_text(encoding="utf-8")

TASK_MODALITY = {"T2T": "text", "TI2T": "text", "T2I": "image", "TI2I": "image"}
TASK_HAS_IMAGE = {"T2T": False, "TI2T": True, "T2I": False, "TI2I": True}

SLOTS = ["interrogative", "interrogative",
         "non-interrogative", "non-interrogative", "non-interrogative"]


def ctx(sample: dict) -> str:
    task = sample["task"]
    return (
        f"Ground-truth output modality: {TASK_MODALITY[task]}\n"
        f"Input image provided: {'yes' if TASK_HAS_IMAGE[task] else 'no'}\n"
    )


# ============================================================ Stage 2a

FIX_SYSTEM = load_prompt("stage2a_fix.txt")

FIX_SCHEMA = {
    "name": "fixed_prompt",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["prompt"],
        "properties": {"prompt": {"type": "string"}},
    },
}


# ============================================================ Stage 2b

GEN_SYSTEM = load_prompt("stage2b_generate.txt")

GEN_SCHEMA = {
    "name": "variants",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["variants"],
        "properties": {
            "variants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["slot", "text"],
                    "properties": {
                        "slot": {"type": "string",
                                 "enum": ["interrogative",
                                          "non-interrogative"]},
                        "text": {"type": "string"},
                    },
                },
            },
        },
    },
}


# ============================================================ Stage 3

VAL_SYSTEM = load_prompt("stage3_validate.txt")

VAL_SCHEMA = {
    "name": "validation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["modality_ok", "meaning_ok", "naturalness_ok", "reason"],
        "properties": {
            "modality_ok": {"type": "boolean"},
            "meaning_ok": {"type": "boolean"},
            "naturalness_ok": {"type": "boolean"},
            "reason": {"type": "string"},
        },
    },
}


# ============================================================ helpers

def is_interrogative(text: str) -> bool:
    return text.strip().endswith("?")


def call(client: OpenAI, model: str, system: str, user: str,
         schema: dict, temperature: float | None) -> dict | None:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(**kwargs)
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if "temperature" in str(e) and "temperature" in kwargs:
                kwargs.pop("temperature")
                continue
            if attempt == 4:
                print(f"\n  FAILED: {e}", file=sys.stderr)
                return None
            time.sleep((2 ** attempt) + random.random())
    return None


# ============================================================ pipeline

def process(client: OpenAI, sample: dict) -> dict:
    rec: dict[str, Any] = {**sample}
    original = sample["prompt"]
    route = sample.get("route", "keep")

    # ---- Stage 2a
    if route == "fix":
        f = call(client, GEN_MODEL, FIX_SYSTEM,
                 ctx(sample) + f'Prompt:\n"""\n{original}\n"""',
                 FIX_SCHEMA, temperature=1.0)
        if f is None or not f.get("prompt"):
            return {**rec, "error": "fix failed", "checks": [], "n_pass": 0}
        source = f["prompt"]
        rec["fixed_prompt"] = source
    else:
        source = original
        rec["fixed_prompt"] = None

    rec["source_prompt"] = source

    # ---- Stage 2b
    gen = call(client, GEN_MODEL, GEN_SYSTEM,
               ctx(sample) + f'Original prompt:\n"""\n{source}\n"""',
               GEN_SCHEMA, temperature=1.0)
    if not gen or not gen.get("variants"):
        return {**rec, "error": "generation failed", "checks": [], "n_pass": 0}

    # ---- Stage 3
    checks = []
    for i, v in enumerate(gen["variants"][: len(SLOTS)]):
        text = v["text"]
        assigned = SLOTS[i]
        c = call(client, VAL_MODEL, VAL_SYSTEM,
                 ctx(sample)
                 + f'Source prompt:\n"""\n{source}\n"""\n'
                 + f'Variant:\n"""\n{text}\n"""',
                 VAL_SCHEMA, temperature=0.0)
        if c is None:
            c = {"modality_ok": False, "meaning_ok": False,
                 "naturalness_ok": False, "reason": "validation failed"}
        c.update({
            "variant": text,
            "slot_assigned": assigned,
            "slot_returned": v["slot"],
            "is_interrogative": is_interrogative(text),
            "slot_ok": is_interrogative(text) == (assigned == "interrogative"),
            "passed": c["modality_ok"] and c["meaning_ok"]
                      and c["naturalness_ok"],
        })
        checks.append(c)

    rec["checks"] = checks
    rec["n_pass"] = sum(c["passed"] for c in checks)
    return rec


# ============================================================ stats

def analyze(results: list[dict]) -> dict[str, Any]:
    rep: dict[str, Any] = {"n_samples": len(results)}
    rep["route"] = dict(Counter(r.get("route", "keep") for r in results))

    all_checks = [c for r in results for c in r.get("checks", [])]
    total = len(all_checks)
    if total == 0:
        return rep

    passed = [c for c in all_checks if c["passed"]]
    rep.update({
        "n_variants_total": total,
        "n_variants_passed": len(passed),
        "variant_pass_rate": round(len(passed) / total, 4),
        "mean_pass": round(
            sum(r.get("n_pass", 0) for r in results) / len(results), 3),
    })

    rep["axis_fail_rate"] = {
        ax: round(sum(1 for c in all_checks if not c[ax]) / total, 4)
        for ax in ("modality_ok", "meaning_ok", "naturalness_ok")
    }

    combos = Counter()
    for c in all_checks:
        if c["passed"]:
            continue
        combos["+".join(ax.replace("_ok", "") for ax in
                        ("modality_ok", "meaning_ok", "naturalness_ok")
                        if not c[ax])] += 1
    rep["fail_combinations"] = dict(combos.most_common())

    dist = Counter(r.get("n_pass", 0) for r in results)
    rep["pass_count_dist"] = {k: dist.get(k, 0) for k in range(len(SLOTS) + 1)}

    rep["slot_compliance"] = round(
        sum(1 for c in all_checks if c["slot_ok"]) / total, 4)

    by_task, by_mod = defaultdict(list), defaultdict(list)
    for r in results:
        m = TASK_MODALITY[r["task"]]
        for c in r.get("checks", []):
            if c["passed"]:
                by_task[r["task"]].append(c["is_interrogative"])
                by_mod[m].append(c["is_interrogative"])
    rep["interrogative_ratio_by_task"] = {
        t: round(sum(v) / len(v), 4) for t, v in sorted(by_task.items()) if v}
    rep["interrogative_ratio_by_modality"] = {
        m: round(sum(v) / len(v), 4) for m, v in by_mod.items() if v}

    route_stat = defaultdict(lambda: [0, 0])
    for r in results:
        st = route_stat[r.get("route", "keep")]
        for c in r.get("checks", []):
            st[0] += 1
            st[1] += int(c["passed"])
    rep["pass_rate_by_route"] = {
        k: {"variants": v[0], "passed": v[1],
            "rate": round(v[1] / v[0], 3) if v[0] else None}
        for k, v in route_stat.items()}

    per_task = defaultdict(list)
    for r in results:
        per_task[r["task"]].append(r.get("n_pass", 0))
    rep["per_task_mean_pass"] = {
        t: round(sum(v) / len(v), 3) for t, v in sorted(per_task.items())}

    rep["low_yield_samples"] = [
        {"id": r.get("id"), "task": r["task"], "route": r.get("route"),
         "prompt": r["prompt"][:60], "n_pass": r.get("n_pass", 0)}
        for r in results if r.get("n_pass", 0) <= 1]

    return rep


# ============================================================ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="stage1_judge.py의 출력 JSON")
    ap.add_argument("--output", default="stage2_results.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 없습니다.", file=sys.stderr)
        return 1

    with open(args.input, encoding="utf-8") as f:
        stage1 = json.load(f)
    samples = [r for r in stage1["results"] if "route" in r]
    if args.limit:
        samples = samples[: args.limit]

    print(f"stage1={args.input} (judge={stage1.get('judge_model')})",
          file=sys.stderr)
    print(f"gen={GEN_MODEL}  val={VAL_MODEL}  n={len(samples)}  "
          f"route={dict(Counter(s['route'] for s in samples))}",
          file=sys.stderr)

    client = OpenAI()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process, client, s) for s in samples]
        for i, fut in enumerate(futures, 1):
            results.append(fut.result())
            print(f"\r  {i}/{len(samples)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    rep = analyze(results)

    print("\n" + "=" * 68)
    print(f"  입력 라우팅: {rep['route']}")
    print(f"  variant 통과율: {rep['n_variants_passed']}/{rep['n_variants_total']} "
          f"({rep['variant_pass_rate']})   평균 {rep['mean_pass']}/{len(SLOTS)}")
    print("  축별 탈락률:")
    for ax, v in rep["axis_fail_rate"].items():
        print(f"    {ax:18s} {v}")
    if rep["fail_combinations"]:
        print(f"  탈락 조합: {rep['fail_combinations']}")
    print("-" * 68)
    print("  샘플당 통과 개수:")
    for k, v in rep["pass_count_dist"].items():
        print(f"    {k}개: {v:3d} {'#' * v}")
    print("-" * 68)
    print(f"  슬롯 준수율: {rep['slot_compliance']}")
    print(f"  통과분 의문형 비율 (모달리티별): "
          f"{rep['interrogative_ratio_by_modality']}")
    print(f"  통과분 의문형 비율 (태스크별):   "
          f"{rep['interrogative_ratio_by_task']}")
    print(f"  경로별 통과율: {rep['pass_rate_by_route']}")
    print(f"  태스크별 평균 통과: {rep['per_task_mean_pass']}")
    print("=" * 68)

    fixed = [r for r in results if r.get("fixed_prompt")]
    if fixed:
        print(f"\n[Stage 2a 수정 {len(fixed)}개 — 앞 10개]")
        for r in fixed[:10]:
            print(f"  {r.get('id')} ({r['task']})")
            print(f"    before: {r['prompt'][:70]}")
            print(f"    after : {r['fixed_prompt'][:70]}")

    if rep.get("low_yield_samples"):
        print(f"\n[통과 1개 이하 {len(rep['low_yield_samples'])}개]")
        for s in rep["low_yield_samples"][:20]:
            print(f"  {s['id']} ({s['task']}/{s['route']}, {s['n_pass']}): "
                  f"{s['prompt']}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"stage1_input": args.input,
                   "judge_model": stage1.get("judge_model"),
                   "gen_model": GEN_MODEL, "val_model": VAL_MODEL,
                   "slots": SLOTS, "n_variants": len(SLOTS),
                   "report": rep, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
