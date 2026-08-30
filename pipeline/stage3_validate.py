"""
Stage 3: 검증 (variant별 modality / meaning / naturalness 3축 검증).

stage2_pipeline.py에 묶여있던 3(검증)만 분리했다. 증강(2b)은 stage2b_generate.py로
따로 뺐다. 시스템 프롬프트(prompts/stage3_validate.txt)는 그대로 재사용.

  입력    stage2b_generate.py의 출력 (results[]에 variants[] 포함, 각 원소는
          {slot_assigned, slot_returned, text, is_interrogative, slot_ok})
  출력    각 샘플에 checks(variant별 modality_ok/meaning_ok/naturalness_ok/reason/
          passed 등)와 n_pass 추가. 나머지는 stage2_pipeline.py의 최종 출력과
          동일한 스키마 -- 리포트(analyze) 로직도 그대로 이식했다.

사용법
------
    python stage3_validate.py --input stage2b_full_5682.json --output stage3_full_5682.json
    python stage3_validate.py --input stage2b_full_5682.json --limit 20 --workers 4
    python stage3_validate.py --input ... --output ... --resume --checkpoint-every 20
    VAL_MODEL=gpt-5.1 python stage3_validate.py ...
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

VAL_MODEL = os.getenv("VAL_MODEL", "gpt-5.5")

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

# ---- 배치 모드: 샘플당 variant 5개를 한 번의 호출로 판정(호출 수 1/5로 감소).
# 비교 실험(200건 파일럿)으로 per-variant 방식과 통과율/판정 품질 차이가 없는지
# 확인한 뒤에만 전량에 쓸 것 -- variant를 한 컨텍스트에 같이 보여주면 모델이
# 서로 비교하며 "채점 곡선"을 그릴 위험이 있어서, 프롬프트에 독립 판정을 강하게
# 명시해뒀다(stage3_validate_batch.txt).
VAL_SYSTEM_BATCH = load_prompt("stage3_validate_batch.txt")

VAL_SCHEMA_BATCH = {
    "name": "validation_batch",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["validations"],
        "properties": {
            "validations": {
                "type": "array",
                "items": {
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
            },
        },
    },
}


def ctx(sample: dict) -> str:
    task = sample["task"]
    return (
        f"Ground-truth output modality: {TASK_MODALITY[task]}\n"
        f"Input image provided: {'yes' if TASK_HAS_IMAGE[task] else 'no'}\n"
    )


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


def process(client: OpenAI, sample: dict) -> dict:
    rec: dict[str, Any] = {**sample}
    source = sample.get("source_prompt", sample.get("prompt"))
    variants = sample.get("variants") or []

    if not variants:
        rec["checks"] = []
        rec["n_pass"] = 0
        return rec

    checks = []
    for v in variants:
        text = v["text"]
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
            "slot_assigned": v.get("slot_assigned"),
            "slot_returned": v.get("slot_returned"),
            "is_interrogative": v.get("is_interrogative"),
            "slot_ok": v.get("slot_ok"),
            "passed": c["modality_ok"] and c["meaning_ok"] and c["naturalness_ok"],
        })
        checks.append(c)

    rec["checks"] = checks
    rec["n_pass"] = sum(c["passed"] for c in checks)
    return rec


def process_batch(client: OpenAI, sample: dict) -> dict:
    """샘플당 1회 호출로 variant 전체를 판정(호출 수 1/5). process()와 같은
    rec["checks"]/rec["n_pass"] 스키마를 만들어서 이후 analyze()가 그대로 먹는다."""
    rec: dict[str, Any] = {**sample}
    source = sample.get("source_prompt", sample.get("prompt"))
    variants = sample.get("variants") or []

    if not variants:
        rec["checks"] = []
        rec["n_pass"] = 0
        return rec

    numbered = "\n".join(f'{i+1}. """{v["text"]}"""' for i, v in enumerate(variants))
    result = call(client, VAL_MODEL, VAL_SYSTEM_BATCH,
                  ctx(sample)
                  + f'Source prompt:\n"""\n{source}\n"""\n\n'
                  + f'Variants ({len(variants)} total):\n{numbered}',
                  VAL_SCHEMA_BATCH, temperature=0.0)

    validations = (result or {}).get("validations") or []
    if len(validations) != len(variants):
        # 개수가 안 맞으면(모델이 몇 개 빼먹었거나 더 만든 경우) 전부 실패 처리 --
        # 잘못 매칭해서 엉뚱한 variant에 판정을 붙이는 것보다 안전.
        checks = [{"modality_ok": False, "meaning_ok": False, "naturalness_ok": False,
                   "reason": f"batch validation count mismatch "
                             f"(got {len(validations)}, expected {len(variants)})",
                   "variant": v["text"], "slot_assigned": v.get("slot_assigned"),
                   "slot_returned": v.get("slot_returned"),
                   "is_interrogative": v.get("is_interrogative"),
                   "slot_ok": v.get("slot_ok"), "passed": False}
                  for v in variants]
    else:
        checks = []
        for v, c in zip(variants, validations):
            c = dict(c)
            c.update({
                "variant": v["text"],
                "slot_assigned": v.get("slot_assigned"),
                "slot_returned": v.get("slot_returned"),
                "is_interrogative": v.get("is_interrogative"),
                "slot_ok": v.get("slot_ok"),
                "passed": c["modality_ok"] and c["meaning_ok"] and c["naturalness_ok"],
            })
            checks.append(c)

    rec["checks"] = checks
    rec["n_pass"] = sum(c["passed"] for c in checks)
    return rec


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
        "mean_pass": round(sum(r.get("n_pass", 0) for r in results) / len(results), 3),
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
         "prompt": r.get("prompt", "")[:60], "n_pass": r.get("n_pass", 0)}
        for r in results if r.get("n_pass", 0) <= 1]

    return rep


def _save(path: str, stage2b_input: str, results: list[dict], rep: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"stage2b_input": stage2b_input, "val_model": VAL_MODEL,
                   "slots": SLOTS, "n_variants": len(SLOTS),
                   "report": rep, "results": results},
                  f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="stage2b_generate.py의 출력 JSON")
    ap.add_argument("--output", default="stage3_results.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--batch-by-sample", action="store_true",
                     help="variant 5개를 샘플당 1회 호출로 묶어서 판정(호출 수 1/5). "
                          "prompts/stage3_validate_batch.txt 사용.")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 없습니다.", file=sys.stderr)
        return 1

    with open(args.input, encoding="utf-8") as f:
        stage2b = json.load(f)
    samples = stage2b["results"]
    if args.limit:
        samples = samples[: args.limit]

    done_results: list[dict] = []
    done_ids: set[str] = set()
    if args.resume and Path(args.output).exists():
        with open(args.output, encoding="utf-8") as f:
            prev = json.load(f)
        done_results = prev.get("results", [])
        done_ids = {r["id"] for r in done_results if "id" in r}
        print(f"[resume] 이미 완료된 {len(done_ids)}건 건너뜀", file=sys.stderr)

    todo = [s for s in samples if s.get("id") not in done_ids]

    print(f"stage2b={args.input}", file=sys.stderr)
    print(f"val={VAL_MODEL}  전체={len(samples)}  남은={len(todo)}", file=sys.stderr)

    client = OpenAI()
    results: list[dict] = list(done_results)
    n_done_this_run = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fn = process_batch if args.batch_by_sample else process
        futures = [ex.submit(fn, client, s) for s in todo]
        for i, fut in enumerate(futures, 1):
            results.append(fut.result())
            n_done_this_run += 1
            print(f"\r  {i}/{len(todo)}", end="", file=sys.stderr)
            if n_done_this_run % args.checkpoint_every == 0:
                _save(args.output, args.input, results, analyze(results))
                print(f"  [checkpoint] {len(results)}건 저장", file=sys.stderr)
    print(file=sys.stderr)

    rep = analyze(results)

    print("\n" + "=" * 68)
    print(f"  입력 라우팅: {rep['route']}")
    if "n_variants_total" in rep:
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
        print(f"  통과분 의문형 비율 (모달리티별): {rep['interrogative_ratio_by_modality']}")
        print(f"  통과분 의문형 비율 (태스크별):   {rep['interrogative_ratio_by_task']}")
        print(f"  경로별 통과율: {rep['pass_rate_by_route']}")
        print(f"  태스크별 평균 통과: {rep['per_task_mean_pass']}")
    print("=" * 68)

    if rep.get("low_yield_samples"):
        print(f"\n[통과 1개 이하 {len(rep['low_yield_samples'])}개]")
        for s in rep["low_yield_samples"][:20]:
            print(f"  {s['id']} ({s['task']}/{s['route']}, {s['n_pass']}): {s['prompt']}")

    _save(args.output, args.input, results, rep)
    print(f"\nsaved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
