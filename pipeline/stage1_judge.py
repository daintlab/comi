"""
Stage 1: 판정 (needs_refinement 버전 — GT 모달리티를 주고 정제 필요 여부 판정).

시스템 프롬프트에 판정 기준의 근거를 넣었다. UMM은 언제나 텍스트로 이어 쓸
수 있으므로 프롬프트가 이미지를 요구하는 것은 이미지를 달라고 했을 때뿐이며,
T2I 데이터셋의 캡션·결과 서술이 이미지 요청으로 읽히는 것은 데이터셋 관습
때문이라는 점을 명시한다.

  judge 입력   프롬프트 + 입력 이미지 동반 여부 + GT 모달리티
  judge 출력   needs_refinement, basis, refine_hint
  코드         needs_refinement true 이면 route=fix

사용법
------
    python stage1_judge.py --input samples.jsonl --output stage1_B.json
    python stage1_judge.py --limit 20 --workers 4
    JUDGE_MODEL=gpt-4o-2024-08-06 python stage1_judge.py
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

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5.5")

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


SYSTEM = load_prompt("stage1_judge.txt")

SCHEMA = {
    "name": "judgement",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["needs_refinement", "basis", "refine_hint"],
        "properties": {
            "needs_refinement": {"type": "boolean"},
            "basis": {"type": "string"},
            "refine_hint": {"type": "string"},
        },
    },
}


def user_msg(sample: dict) -> str:
    task = sample["task"]
    return (
        f"Ground-truth output modality: {TASK_MODALITY[task]}\n"
        f"Input image provided: {'yes' if TASK_HAS_IMAGE[task] else 'no'}\n"
        f'Prompt:\n"""\n{sample["prompt"]}\n"""'
    )


def call(client: OpenAI, sample: dict) -> dict | None:
    kwargs: dict[str, Any] = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user_msg(sample)}],
        "response_format": {"type": "json_schema", "json_schema": SCHEMA},
        "temperature": 0.0,
    }
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


def analyze(results: list[dict]) -> dict[str, Any]:
    ok = [r for r in results if "route" in r]
    rep: dict[str, Any] = {"n_samples": len(results), "n_judged": len(ok)}
    if not ok:
        return rep

    rep["gate"] = dict(Counter(r["route"] for r in ok))
    rep["fix_rate"] = round(
        sum(1 for r in ok if r["route"] == "fix") / len(ok), 4)

    per_task = defaultdict(Counter)
    for r in ok:
        per_task[r["task"]][r["route"]] += 1
    rep["gate_per_task"] = {
        t: {**dict(c), "fix_rate": round(c["fix"] / sum(c.values()), 3)}
        for t, c in sorted(per_task.items())
    }

    per_note = defaultdict(Counter)
    for r in ok:
        per_note[r.get("note", "")][r["route"]] += 1
    rep["gate_per_note"] = {n: dict(c) for n, c in sorted(per_note.items())}
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="samples.jsonl")
    ap.add_argument("--output", default="stage1_B.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 없습니다.", file=sys.stderr)
        return 1

    with open(args.input, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        samples = samples[: args.limit]

    print(f"judge={JUDGE_MODEL}  n={len(samples)}", file=sys.stderr)
    client = OpenAI()

    def work(s: dict) -> dict:
        j = call(client, s)
        if j is None:
            return {**s, "error": "judge failed"}
        return {**s, **j,
                "gt_modality": TASK_MODALITY[s["task"]],
                "route": "fix" if j["needs_refinement"] else "keep"}

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, s) for s in samples]
        for i, fut in enumerate(futures, 1):
            results.append(fut.result())
            print(f"\r  {i}/{len(samples)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    rep = analyze(results)

    print("\n" + "=" * 68)
    print(f"  게이트: {rep.get('gate')}   fix 비율 {rep.get('fix_rate')}")
    print("-" * 68)
    print("  태스크별:")
    for t, v in rep.get("gate_per_task", {}).items():
        print(f"    {t:6s} keep {v.get('keep', 0):3d}  fix {v.get('fix', 0):3d}"
              f"  ({v['fix_rate']})")
    print("-" * 68)
    print("  note별:")
    for n, v in rep.get("gate_per_note", {}).items():
        print(f"    {n[:34]:34s} keep {v.get('keep', 0):3d}  "
              f"fix {v.get('fix', 0):3d}")
    print("=" * 68)

    fixes = [r for r in results if r.get("route") == "fix"]
    print(f"\n[fix로 라우팅된 {len(fixes)}개]")
    for r in fixes:
        print(f"\n  {r.get('id')} ({r['task']}, {r.get('note','')})")
        print(f"    prompt: {r['prompt'][:70]}")
        print(f"    basis : {r['basis'][:110]}")
        print(f"    hint  : {r['refine_hint'][:110]}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"judge_model": JUDGE_MODEL, "method": "needs_refinement",
                   "report": rep, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
