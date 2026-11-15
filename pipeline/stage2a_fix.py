"""
Stage 2a: 수정 (fix).

stage2_pipeline.py에 묶여있던 2a(수정)만 분리했다. route == "fix"인 샘플만
재작성하고, keep은 원본 그대로 통과시킨다. 시스템 프롬프트(prompts/stage2a_fix.txt)는
그대로 재사용 — 건드리지 않았다.

Stage 2a는 stage1_judge.py의 basis/refine_hint를 사용자 메시지에 그대로 넣는다.
원래는 hint를 안 썼는데(judge 판정 근거가 그대로 처방이 되어 모든 프롬프트에
"Generate an image:"가 붙을 거라는 우려 때문), 실제로 30건 ablation(결과:
results/hint_ablation_30.md)을 돌려보니 반대였다 -- hint 없이 자유 재작성하면
오히려 동작/디테일이 드리프트하는 경우(예: "extract" -> "draw", 입력 이미지
편집 요청이 신규 생성 요청으로 바뀜)가 많았고, meaning_ok 통과율이 23/30(힌트
없음)보다 29/30(힌트 있음)으로 더 높았다. 그래서 hint를 쓰는 쪽으로 바꿨다.

  입력    stage1_judge.py의 출력 (results[]에 route, needs_refinement 포함)
  출력    각 샘플에 fixed_prompt(route=="fix"일 때만 값 있음, 아니면 None)와
          source_prompt(2b/3에서 실제로 쓸 프롬프트 -- fix면 수정본, keep이면 원본)
          를 추가. 이후 stage2_pipeline.py(2b/3)가 이 출력을 입력으로 받으면
          source_prompt가 이미 있으므로 2a를 다시 돌리지 않는다.

사용법
------
    python stage2a_fix.py --input stage1_full_5682.json --output stage2a_full_5682.json
    python stage2a_fix.py --input stage1_full_5682.json --limit 20 --workers 4
    GEN_MODEL=gpt-5.5 python stage2a_fix.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GEN_MODEL = os.getenv("GEN_MODEL", "gpt-5.5")

# 시스템 프롬프트는 prompts/ 아래 txt로 관리한다. 스크립트를 건드리지 않고
# 프롬프트만 수정할 수 있게 하기 위함. (stage1_judge.py / stage2_pipeline.py와 동일 관례)
PROMPT_DIR = Path(os.getenv("PROMPT_DIR", "prompts"))


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    if not path.exists():
        print(f"프롬프트 파일이 없습니다: {path}", file=sys.stderr)
        raise SystemExit(1)
    return path.read_text(encoding="utf-8")


TASK_MODALITY = {"T2T": "text", "TI2T": "text", "T2I": "image", "TI2I": "image"}
TASK_HAS_IMAGE = {"T2T": False, "TI2T": True, "T2I": False, "TI2I": True}

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
    original = sample["prompt"]
    route = sample.get("route", "keep")

    if route == "fix":
        user = (ctx(sample)
                + f'Diagnosis: {sample["basis"]}\n'
                + f'Suggestion: {sample["refine_hint"]}\n'
                + f'Prompt:\n"""\n{original}\n"""')
        f = call(client, GEN_MODEL, FIX_SYSTEM, user, FIX_SCHEMA, temperature=1.0)
        if f is None or not f.get("prompt"):
            return {**rec, "fixed_prompt": None, "source_prompt": original,
                    "error": "fix failed"}
        rec["fixed_prompt"] = f["prompt"]
        rec["source_prompt"] = f["prompt"]
    else:
        rec["fixed_prompt"] = None
        rec["source_prompt"] = original

    return rec


def analyze(results: list[dict]) -> dict[str, Any]:
    rep: dict[str, Any] = {"n_samples": len(results)}
    rep["route"] = dict(Counter(r.get("route", "keep") for r in results))

    fix_targets = [r for r in results if r.get("route") == "fix"]
    n_failed = sum(1 for r in fix_targets if r.get("error") == "fix failed")
    rep["n_fix_targets"] = len(fix_targets)
    rep["n_fix_succeeded"] = len(fix_targets) - n_failed
    rep["n_fix_failed"] = n_failed

    by_note = Counter()
    for r in fix_targets:
        by_note[r.get("note", "")] += 1
    rep["fix_by_note"] = dict(by_note)

    by_task = Counter(r["task"] for r in fix_targets)
    rep["fix_by_task"] = dict(by_task)

    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="stage1_judge.py의 출력 JSON")
    ap.add_argument("--output", default="stage2a_results.json")
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

    print(f"stage1={args.input} (judge={stage1.get('judge_model')})", file=sys.stderr)
    print(f"gen={GEN_MODEL}  n={len(samples)}  "
          f"route={dict(Counter(s['route'] for s in samples))}", file=sys.stderr)

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
    print(f"  라우팅: {rep['route']}")
    print(f"  fix 대상: {rep['n_fix_targets']}  성공: {rep['n_fix_succeeded']}  "
          f"실패: {rep['n_fix_failed']}")
    print(f"  note별 fix 대상: {rep['fix_by_note']}")
    print(f"  task별 fix 대상: {rep['fix_by_task']}")
    print("=" * 68)

    fixed = [r for r in results if r.get("fixed_prompt")]
    if fixed:
        print(f"\n[수정 {len(fixed)}개 — 앞 10개]")
        for r in fixed[:10]:
            print(f"  {r.get('id')} ({r['task']})")
            print(f"    before: {r['prompt'][:70]}")
            print(f"    after : {r['fixed_prompt'][:70]}")

    failed = [r for r in results if r.get("error") == "fix failed"]
    if failed:
        print(f"\n[수정 실패 {len(failed)}개]")
        for r in failed[:10]:
            print(f"  {r.get('id')} ({r['task']}): {r['prompt'][:70]}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"stage1_input": args.input,
                   "judge_model": stage1.get("judge_model"),
                   "gen_model": GEN_MODEL,
                   "report": rep, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
