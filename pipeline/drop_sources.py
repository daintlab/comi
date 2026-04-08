#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_drop_list(xs: List[str]) -> Set[str]:
    out: Set[str] = set()
    for x in xs:
        if isinstance(x, str) and x.strip():
            out.add(x.strip())
    return out


def recompute_datasets(results: List[Dict[str, Any]]) -> List[str]:
    seen: Set[str] = set()
    datasets: List[str] = []
    for r in results:
        src = r.get("source")
        if isinstance(src, str) and src not in seen:
            seen.add(src)
            datasets.append(src)
    return datasets


def drop_sources(data: Dict[str, Any], drop: Set[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(data, dict):
        raise ValueError("input_json must be a JSON object (dict)")
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("input_json must contain results[]")

    kept: List[Dict[str, Any]] = []
    dropped = 0
    by_source_dropped: Dict[str, int] = {}

    for r in results:
        if not isinstance(r, dict):
            continue
        src = r.get("source")
        if isinstance(src, str) and src in drop:
            dropped += 1
            by_source_dropped[src] = by_source_dropped.get(src, 0) + 1
            continue
        kept.append(r)

    data["results"] = kept
    data["total_questions"] = len(kept)
    data["datasets"] = recompute_datasets(kept)

    stats = {"dropped_total": dropped, "dropped_by_source": dict(sorted(by_source_dropped.items()))}
    return data, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Drop items from results[] by source name(s).")
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--drop_source", action="append", default=[], help="Source name to drop (repeatable).")
    args = ap.parse_args()

    data = load_json(args.input_json)
    drop = normalize_drop_list(args.drop_source)
    out, stats = drop_sources(data, drop)
    save_json(args.output_json, out)
    print(json.dumps({"output_json": args.output_json, "stats": stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()

