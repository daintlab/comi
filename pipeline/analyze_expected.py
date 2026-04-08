#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--mode", choices=["stacked", "grouped"], default="stacked")
    args = ap.parse_args()

    obj = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    rows = obj.get("results", [])
    if not isinstance(rows, list):
        raise ValueError("input_json must contain top-level key 'results' as a list")

    df = pd.DataFrame(rows)
    if "source" not in df.columns or "expected" not in df.columns:
        raise ValueError("results[] must contain 'source' and 'expected' fields")

    # expected 라벨 정리: expected / not_expected 외는 other로
    df["expected"] = df["expected"].fillna("other").astype(str)
    df.loc[~df["expected"].isin(["expected", "not_expected"]), "expected"] = "other"

    counts = (
        df.groupby(["source", "expected"])
        .size()
        .unstack(fill_value=0)
    )

    # 컬럼 보강 + 순서 고정
    for col in ["expected", "not_expected", "other"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["expected", "not_expected", "other"]]

    # 구버전 pandas 호환: errors= 인자 없이 정렬
    sort_cols = [c for c in ["expected", "not_expected", "other"] if c in counts.columns]
    if sort_cols:
        counts = counts.sort_values(by=sort_cols, ascending=False)

    # 출력 폴더 생성
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    ax = plt.gca()

    if args.mode == "stacked":
        counts.plot(kind="bar", stacked=True, ax=ax)
        plt.ylabel("count (stacked)")
    else:
        counts[["expected", "not_expected"]].plot(kind="bar", stacked=False, ax=ax)
        plt.ylabel("count")

    plt.title("expected vs not_expected by source")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"saved: {args.out_png}")


if __name__ == "__main__":
    main()