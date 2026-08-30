"""
Stage 2b: 증강 (variant 생성만).

stage2_pipeline.py에 묶여있던 2b(증강)만 분리했다. 검증(3)은 stage3_validate.py로
따로 뺐다. 시스템 프롬프트(prompts/stage2b_generate.txt)는 그대로 재사용.

  입력    stage2a_fix.py의 출력 (results[]에 source_prompt/fixed_prompt 포함).
          stage1_judge.py의 출력을 바로 넣어도 동작한다(그 경우 source_prompt가
          없으니 route=="fix"면 원본 prompt를 그대로 source로 씀 -- 2a를 안 거친
          채로 생성하게 되므로 원칙적으로는 stage2a_fix.py를 먼저 돌리는 걸 권장).
  출력    각 샘플에 variants(슬롯 순서대로 5개, {slot, text}) 추가.
          검증은 안 하므로 checks/n_pass는 여기 없음 -- stage3_validate.py가 채움.

사용법
------
    python stage2b_generate.py --input stage2a_full_5682.json --output stage2b_full_5682.json
    python stage2b_generate.py --input stage2a_full_5682.json --limit 20 --workers 4
    python stage2b_generate.py --input ... --output ... --resume --checkpoint-every 20
    GEN_MODEL=gpt-5.1 python stage2b_generate.py ...
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
                                 "enum": ["interrogative", "non-interrogative"]},
                        "text": {"type": "string"},
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


def process(client: OpenAI, sample: dict) -> dict:
    rec: dict[str, Any] = {**sample}
    source = sample.get("source_prompt")
    if source is None:
        # stage2a_fix.py를 안 거치고 stage1 출력을 바로 받은 경우의 fallback
        source = sample["prompt"]
        rec["source_prompt"] = source
        rec.setdefault("fixed_prompt", None)

    gen = call(client, GEN_MODEL, GEN_SYSTEM,
               ctx(sample) + f'Original prompt:\n"""\n{source}\n"""',
               GEN_SCHEMA, temperature=1.0)
    if not gen or not gen.get("variants"):
        return {**rec, "variants": [], "error": "generation failed"}

    variants = []
    for i, v in enumerate(gen["variants"][: len(SLOTS)]):
        variants.append({
            "slot_assigned": SLOTS[i],
            "slot_returned": v["slot"],
            "text": v["text"],
            "is_interrogative": is_interrogative(v["text"]),
            "slot_ok": is_interrogative(v["text"]) == (SLOTS[i] == "interrogative"),
        })
    rec["variants"] = variants
    return rec


def analyze(results: list[dict]) -> dict[str, Any]:
    rep: dict[str, Any] = {"n_samples": len(results)}
    rep["route"] = dict(Counter(r.get("route", "keep") for r in results))

    failed = [r for r in results if r.get("error") == "generation failed"]
    rep["n_generation_failed"] = len(failed)

    ok = [r for r in results if r.get("variants")]
    rep["n_generated"] = len(ok)
    rep["mean_variants_per_sample"] = round(
        sum(len(r["variants"]) for r in ok) / len(ok), 3) if ok else 0

    all_variants = [v for r in ok for v in r["variants"]]
    if all_variants:
        rep["slot_compliance"] = round(
            sum(1 for v in all_variants if v["slot_ok"]) / len(all_variants), 4)

    by_task = Counter(r["task"] for r in results)
    rep["by_task"] = dict(by_task)

    return rep


def _save(path: str, stage2a_input: str, results: list[dict], rep: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"stage2a_input": stage2a_input, "gen_model": GEN_MODEL,
                   "slots": SLOTS, "n_variants": len(SLOTS),
                   "report": rep, "results": results},
                  f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="stage2a_fix.py의 출력 JSON")
    ap.add_argument("--output", default="stage2b_results.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 없습니다.", file=sys.stderr)
        return 1

    with open(args.input, encoding="utf-8") as f:
        stage2a = json.load(f)
    samples = stage2a["results"]
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

    print(f"stage2a={args.input}", file=sys.stderr)
    print(f"gen={GEN_MODEL}  전체={len(samples)}  남은={len(todo)}", file=sys.stderr)

    client = OpenAI()
    results: list[dict] = list(done_results)
    n_done_this_run = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process, client, s) for s in todo]
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
    print(f"  라우팅: {rep['route']}")
    print(f"  생성 실패: {rep['n_generation_failed']}  성공: {rep['n_generated']}")
    print(f"  샘플당 평균 variant 수: {rep['mean_variants_per_sample']}")
    if "slot_compliance" in rep:
        print(f"  슬롯 준수율(생성 단계, 검증 전): {rep['slot_compliance']}")
    print(f"  태스크별: {rep['by_task']}")
    print("=" * 68)

    _save(args.output, args.input, results, rep)
    print(f"\nsaved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
