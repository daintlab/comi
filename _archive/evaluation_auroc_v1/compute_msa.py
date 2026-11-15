#!/usr/bin/env python3
"""
Modality Selection Accuracy (MSA) 평가 — B-MSA / hard-MSA / soft-MSA.

입력 형식:
  (1) 디렉터리 (권장): item_XXXX/ 또는 batch_XXXX/item_XXXX/ 구조
      각 item: meta.json + output.json (또는 output_r00.json ... output_rNN.json)
  (2) 단일 JSON (레거시): {results: [{model_outputs: [...]}]}

태스크 타입 매핑 (source → expected modality):
  t2t:  gsm8k, mt-bench-101        → 텍스트 출력
  ti2t: InfoVQA, VQAv2             → 텍스트 출력
  t2i:  geneval, dpg-bench         → 이미지 출력
  ti2i: EditBench, ImgEditBench    → 이미지 출력

지표:
  B-MSA:    Text macro-MSA(t2t+ti2t)와 Image macro-MSA(t2i+ti2i)의 조화평균
  hard-MSA: 샘플당 모든 variant가 정답인 비율
  soft-MSA: 샘플당 1개 이상 정답인 비율

사용 예:
  python compute_msa.py -i /path/to/anole_inference/
  python compute_msa.py -i /path/to/msa_repeat2/batch_0000/ --repeat-agg any
  python compute_msa.py -i /path/to/legacy.json
"""
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

SOURCE_TO_TASK = {
    "gsm8k": "t2t",
    "mt-bench-101": "t2t",
    "mmlu": "t2t",
    "InfoVQA": "ti2t",
    "VQAv2": "ti2t",
    "ScienceQA": "ti2t",
    "geneval": "t2i",
    "dpg-bench": "t2i",
    "dpg_bench": "t2i",
    "EditBench": "ti2i",
    "ImgEditBench": "ti2i",
    "MagicBrush": "ti2i",
}

TASK_TO_MODALITY = {
    "t2t": "text",
    "ti2t": "text",
    "t2i": "image",
    "ti2i": "image",
}

IMAGE_TOKEN_RE = re.compile(r"<I\d+>")


# ---------------------------------------------------------------------------
# Output modality detection
# ---------------------------------------------------------------------------

def output_has_image_dir(segments: list[dict]) -> bool:
    """디렉터리 형식: output segments 중 type='image'가 있으면 True."""
    return any(seg.get("type") == "image" for seg in segments)


def output_has_text_dir(segments: list[dict]) -> bool:
    """디렉터리 형식: output segments 중 비어있지 않은 text가 있으면 True."""
    for seg in segments:
        if seg.get("type") == "text":
            content = seg.get("content", "")
            if isinstance(content, str) and content.strip():
                return True
    return False


def output_has_image_token(text: str) -> bool:
    """레거시(Tar) 형식: <I숫자> 토큰이 있으면 True."""
    if not text or not isinstance(text, str):
        return False
    return bool(IMAGE_TOKEN_RE.search(text))


def is_correct(has_image: bool, has_text: bool, task_type: str) -> bool:
    if task_type in ("t2t", "ti2t"):
        return has_text               # 텍스트 답변이 있으면 정답 (이미지 포함 무관)
    if task_type in ("t2i", "ti2i"):
        return has_image              # 이미지가 있어야 정답
    return False


# ---------------------------------------------------------------------------
# Directory loader
# ---------------------------------------------------------------------------

def _find_item_dirs(root: Path) -> list[Path]:
    """item_XXXX 디렉터리를 찾는다. flat / batch 구조 모두 지원."""
    items = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("item_"):
            items.append(entry)
        elif entry.name.startswith("batch_"):
            for sub in sorted(entry.iterdir()):
                if sub.is_dir() and sub.name.startswith("item_"):
                    items.append(sub)
    return items


def _load_output_segments(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _output_is_empty(segments: list[dict]) -> bool:
    """output segments가 실질적으로 비어있으면 True."""
    if not segments:
        return True
    for seg in segments:
        if seg.get("type") == "image":
            return False
        content = seg.get("content", "")
        if isinstance(content, str) and content.strip():
            return False
    return True


def load_items_from_dir(run_dir: str, repeat_agg: str = "any") -> list[dict]:
    """디렉터리에서 item을 로드하여 variant별 correct 판정을 반환.

    repeat_agg: repeat가 여러 개일 때 집계 방식
      - "first": 첫 번째 repeat(r00)만 사용
      - "any":   하나라도 정답이면 correct (default)
      - "all":   모든 repeat가 정답이어야 correct
      - "majority": 과반수가 정답이면 correct
    """
    root = Path(run_dir)
    item_dirs = _find_item_dirs(root)
    if not item_dirs:
        return []

    items = []
    for item_dir in item_dirs:
        meta_path = item_dir / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        task_id = meta.get("task_id", "")
        source = meta.get("source", "")
        task_type = SOURCE_TO_TASK.get(source)
        if task_type is None:
            continue

        # output 파일 수집
        output_path = item_dir / "output.json"
        repeat_paths = sorted(item_dir.glob("output_r*.json"))

        if output_path.exists() and not repeat_paths:
            segments = _load_output_segments(output_path)
            empty = _output_is_empty(segments)
            has_img = output_has_image_dir(segments)
            has_txt = output_has_text_dir(segments)
            correct = False if empty else is_correct(has_img, has_txt, task_type)
        elif repeat_paths:
            repeat_corrects = []
            for rp in repeat_paths:
                segments = _load_output_segments(rp)
                empty = _output_is_empty(segments)
                has_img = output_has_image_dir(segments)
                has_txt = output_has_text_dir(segments)
                repeat_corrects.append(False if empty else is_correct(has_img, has_txt, task_type))

            if repeat_agg == "first":
                correct = repeat_corrects[0]
            elif repeat_agg == "all":
                correct = all(repeat_corrects)
            elif repeat_agg == "majority":
                correct = sum(repeat_corrects) > len(repeat_corrects) / 2
            else:  # "any"
                correct = any(repeat_corrects)
        else:
            continue

        items.append({
            "task_id": task_id,
            "source": source,
            "task_type": task_type,
            "correct": correct,
        })

    return items


# ---------------------------------------------------------------------------
# Legacy JSON loader
# ---------------------------------------------------------------------------

def load_items_from_json(json_path: str) -> list[dict]:
    """레거시 단일 JSON ({results: [...]}) 로더."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    items = []
    for r in results:
        source = r.get("source", "")
        task_type = SOURCE_TO_TASK.get(source)
        if task_type is None:
            continue

        # model_outputs (리스트) 또는 model_output (단일)
        outputs = r.get("model_outputs", [])
        if not outputs and "model_output" in r:
            outputs = [r["model_output"]]

        task_id = r.get("task_id", "")
        for vi, out in enumerate(outputs):
            empty = out is None or (isinstance(out, str) and not out.strip())
            has_img = output_has_image_token(out) if isinstance(out, str) else False
            has_txt = isinstance(out, str) and bool(out.strip())
            correct = False if empty else is_correct(has_img, has_txt, task_type)
            items.append({
                "task_id": f"{task_id}::v{vi}",
                "source": source,
                "task_type": task_type,
                "correct": correct,
            })

    return items


# ---------------------------------------------------------------------------
# MSA 계산
# ---------------------------------------------------------------------------

def harmonic_mean(a: float, b: float) -> float:
    if a + b == 0:
        return 0.0
    return 2 * a * b / (a + b)


def _summarize(samples: list[dict]) -> dict:
    if not samples:
        return {"macro_msa": None, "hard_msa": None, "soft_msa": None, "n_samples": 0}
    return {
        "macro_msa": sum(s["macro"] for s in samples) / len(samples),
        "hard_msa": sum(s["hard"] for s in samples) / len(samples),
        "soft_msa": sum(s["soft"] for s in samples) / len(samples),
        "n_samples": len(samples),
    }


def compute_msa(input_path: str, repeat_agg: str = "any", save_json_path: str | None = None) -> dict:
    # --- 로드 ---
    if os.path.isdir(input_path):
        items = load_items_from_dir(input_path, repeat_agg=repeat_agg)
    else:
        items = load_items_from_json(input_path)

    if not items:
        return {"error": "No valid items found"}

    # --- base_task_id 기준으로 variant 그룹핑 ---
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        base_id = item["task_id"].rsplit("::", 1)[0]
        groups[base_id].append(item)

    by_task: dict[str, list] = defaultdict(list)
    for base_id, variants in groups.items():
        source = variants[0]["source"]
        task_type = variants[0]["task_type"]
        corrects = [v["correct"] for v in variants]
        n = len(corrects)

        by_task[task_type].append({
            "task_id": base_id,
            "source": source,
            "n_variants": n,
            "macro": sum(corrects) / n if n else 0.0,
            "hard": 1.0 if all(corrects) else 0.0,
            "soft": 1.0 if any(corrects) else 0.0,
            "variant_corrects": corrects,
        })

    # --- 태스크별 집계 ---
    task_summary = {tt: _summarize(by_task.get(tt, [])) for tt in ("t2t", "ti2t", "t2i", "ti2i")}

    # --- modality별 집계 ---
    modality_samples: dict[str, list] = defaultdict(list)
    for tt, samples in by_task.items():
        modality_samples[TASK_TO_MODALITY[tt]].extend(samples)
    modality_summary = {mod: _summarize(modality_samples.get(mod, [])) for mod in ("text", "image")}

    # --- 전체 집계 ---
    all_samples = [s for samples in by_task.values() for s in samples]
    n_all = len(all_samples)
    overall = {
        "hard_msa": sum(s["hard"] for s in all_samples) / n_all if n_all else None,
        "soft_msa": sum(s["soft"] for s in all_samples) / n_all if n_all else None,
        "n_samples": n_all,
    }

    # --- B-MSA ---
    text_macro = modality_summary["text"]["macro_msa"]
    image_macro = modality_summary["image"]["macro_msa"]
    b_msa = harmonic_mean(text_macro, image_macro) if text_macro is not None and image_macro is not None else None

    out = {
        "input": input_path,
        "repeat_agg": repeat_agg if os.path.isdir(input_path) else None,
        "headline": {
            "B-MSA": b_msa,
            "hard-MSA": overall["hard_msa"],
            "soft-MSA": overall["soft_msa"],
        },
        "modality": modality_summary,
        "by_task": task_summary,
        "overall": overall,
        "per_sample_detail": {
            tt: [
                {
                    "task_id": s["task_id"],
                    "source": s["source"],
                    "n_variants": s["n_variants"],
                    "macro": s["macro"],
                    "hard": s["hard"],
                    "soft": s["soft"],
                    "variant_corrects": s["variant_corrects"],
                }
                for s in samples
            ]
            for tt, samples in by_task.items()
        },
    }

    if save_json_path:
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved MSA report → {save_json_path}")

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(result: dict):
    h = result["headline"]
    W = 64
    print("\n" + "=" * W)
    print("  AugBench MSA Report")
    print("=" * W)

    if result.get("repeat_agg"):
        print(f"  (repeat aggregation: {result['repeat_agg']})")

    print()
    print(f"  {'Metric':<12} {'Score':>10}")
    print(f"  {'-'*12} {'-'*10}")
    print(f"  {'B-MSA':<12} {h['B-MSA']:>10.4f}" if h["B-MSA"] is not None else f"  {'B-MSA':<12} {'N/A':>10}")
    print(f"  {'hard-MSA':<12} {h['hard-MSA']:>10.4f}" if h["hard-MSA"] is not None else f"  {'hard-MSA':<12} {'N/A':>10}")
    print(f"  {'soft-MSA':<12} {h['soft-MSA']:>10.4f}" if h["soft-MSA"] is not None else f"  {'soft-MSA':<12} {'N/A':>10}")

    print(f"\n  {'--- Modality-level ---':^{W-4}}")
    print(f"  {'Modality':<10} {'N':>5} {'macro':>8} {'hard':>8} {'soft':>8}")
    print(f"  {'-'*10} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
    for mod in ("text", "image"):
        m = result["modality"].get(mod, {})
        n = m.get("n_samples", 0)
        if n == 0:
            print(f"  {mod:<10} {'--':>5} {'--':>8} {'--':>8} {'--':>8}")
            continue
        print(f"  {mod:<10} {n:>5} {m['macro_msa']:>8.4f} {m['hard_msa']:>8.4f} {m['soft_msa']:>8.4f}")

    print(f"\n  {'--- Task-level breakdown ---':^{W-4}}")
    print(f"  {'Task':<10} {'N':>5} {'macro':>8} {'hard':>8} {'soft':>8}")
    print(f"  {'-'*10} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
    for tt in ("t2t", "ti2t", "t2i", "ti2i"):
        s = result["by_task"].get(tt, {})
        n = s.get("n_samples", 0)
        if n == 0:
            print(f"  {tt:<10} {'--':>5} {'--':>8} {'--':>8} {'--':>8}")
            continue
        print(f"  {tt:<10} {n:>5} {s['macro_msa']:>8.4f} {s['hard_msa']:>8.4f} {s['soft_msa']:>8.4f}")

    # Source-level breakdown
    detail = result.get("per_sample_detail", {})
    if detail:
        from collections import defaultdict
        source_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for tt, samples in detail.items():
            for s in samples:
                src = s["source"]
                source_stats[src]["correct"] += s["macro"]
                source_stats[src]["total"] += 1
                source_stats[src]["task_type"] = tt

        print(f"\n  {'--- Source-level breakdown ---':^{W-4}}")
        print(f"  {'Source':<16} {'Task':>5} {'N':>5} {'Accuracy':>10}")
        print(f"  {'-'*16} {'-'*5} {'-'*5} {'-'*10}")
        for src in sorted(source_stats.keys()):
            st = source_stats[src]
            acc = st["correct"] / st["total"] if st["total"] else 0
            print(f"  {src:<16} {st['task_type']:>5} {st['total']:>5} {acc:>10.4f}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compute B-MSA, hard-MSA, soft-MSA from model inference results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s -i /path/to/anole_inference/
  %(prog)s -i /path/to/run_dir/ --repeat-agg majority
  %(prog)s -i /path/to/legacy.json
""",
    )
    parser.add_argument("--input", "-i", required=True,
                        help="추론 결과 디렉터리 (item_XXXX/) 또는 레거시 JSON 파일")
    parser.add_argument("--output", "-o", default=None,
                        help="MSA 리포트 JSON 저장 경로")
    parser.add_argument("--repeat-agg", default="any", choices=["first", "any", "all", "majority"],
                        help="repeat 집계 방식 (디렉터리 입력 시, default: any)")
    args = parser.parse_args()

    result = compute_msa(args.input, repeat_agg=args.repeat_agg, save_json_path=args.output)

    if "error" in result:
        print(result["error"])
        return

    _print_report(result)


if __name__ == "__main__":
    main()
