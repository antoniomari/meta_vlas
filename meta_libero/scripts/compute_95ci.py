#!/usr/bin/env python3
"""Group TTT results by hyperparameters and compute 95% Gaussian CI.

Example:
  python meta_libero/scripts/compute_95ci.py \
    meta_libero/results/ttt_new/dataset_libero_90_reptile/lora/results_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


DEFAULT_EXCLUDE = {"seed", "success_rate"}


def _gaussian_ci(values: list[float]) -> tuple[float, float, float, float]:
    n = len(values)
    mean = statistics.fmean(values)
    if n <= 1:
        return mean, mean, mean, 0.0
    s = statistics.stdev(values)
    se = s / math.sqrt(n)
    half = 1.96 * se
    return mean, mean - half, mean + half, half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", help="Path to results_summary.csv")
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=None,
        help="Optional explicit grouping columns. Default: merging_eps.",
    )
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        print("No rows found.")
        return

    if args.group_by:
        group_cols = list(args.group_by)
    else:
        group_cols = ["merging_eps"]

    missing = [c for c in group_cols if c not in fieldnames]
    if missing:
        raise ValueError(f"Unknown grouping columns: {missing}. Available: {fieldnames}")

    groups: dict[tuple[str, ...], list[float]] = {}
    for row in rows:
        sr = row.get("success_rate", "")
        if sr == "":
            continue
        key = tuple(row.get(c, "") for c in group_cols)
        groups.setdefault(key, []).append(float(sr))

    # stable sorted output
    sorted_items = sorted(groups.items(), key=lambda kv: kv[0])
    print(f"Loaded {len(rows)} rows from: {csv_path}")
    print(f"Grouping by: {', '.join(group_cols)}")
    print("-" * 100)

    for key, values in sorted_items:
        mean, lo, hi, half = _gaussian_ci(values)
        group_desc = ", ".join(f"{c}={v}" for c, v in zip(group_cols, key))
        print(
            f"{group_desc} | n={len(values)} | "
            f"{mean:.2f} ± {half:.2f} | 95% CI=[{lo:.4f}, {hi:.4f}]"
        )


if __name__ == "__main__":
    main()
