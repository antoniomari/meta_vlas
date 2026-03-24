#!/usr/bin/env python3
"""
Aggregate augment_finetune results into a single CSV.

Given an input folder of results (with subfolders like task7_task6_seed42_onpolicy_self_replay/lr5e-05_steps200_eps1.0/),
creates aggregate_results.csv in the main folder with metrics per hyperparameter combination:
- task1_success_rate (phase 2 final)
- task2_success_rate (phase 2 final)
- avg_success_rate
- auc_task1 (normalized area under curve)
- auc_task2 (normalized area under curve)
- harmonic_mean_success_rate (2*s1*s2/(s1+s2) for task1/task2 final SR)

Also writes aggregate_results_general.csv: same metrics averaged over all (task1, task2) pairs per setting (mode, lr, steps, eps, seed, optional align/warmup/lambda_kl).

Usage:
  python aggregate_augment_finetune_results.py /path/to/results/folder
  python aggregate_augment_finetune_results.py results/augment_finetune
  python aggregate_augment_finetune_results.py results/augment_finetune --steps 200  # only process runs with 200 steps
"""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


def parse_task_folder(name: str) -> dict | None:
    """Parse task folder like task7_task6_seed42_onpolicy_self_replay or task4_task5_seed42_noaugment."""
    # task{T1}_task{T2}_seed{seed} or task{T1}_task{T2}_seed{seed}_suffix
    m = re.match(r"task(\d+)_task(\d+)_seed(\d+)(?:_(.+))?$", name)
    if not m:
        return None
    task1, task2, seed, suffix = m.groups()
    mode = suffix if suffix else "self_replay"
    return {"task1": int(task1), "task2": int(task2), "seed": int(seed), "mode": mode}


def parse_hyperparam_folder(name: str) -> dict | None:
    """Parse hyperparam folder like lr5e-05_steps200_eps1.0 or lr2.5e-05_steps200_eps0.5_align0.2_warmup20_lambdaKL0.5.

    If _warmup{N} is omitted, warmup_steps defaults to 0. If _lambdaKL{V} is omitted, lambda_kl defaults to 1.0.
    """
    # lr{X}_steps{N}_eps{E}[_align{A}][_warmup{N}][_lambdaKL{V}]
    m = re.match(r"lr([\d.e+-]+)_steps(\d+)_eps([\d.]+)(?:_align([\d.]+))?(?:_warmup(\d+))?(?:_lambdaKL([\d.]+))?$", name)
    if not m:
        return None
    lr_str, steps, eps, align, warmup, lambda_kl = m.groups()
    # Parse LR: 5e-05, 2.5e-05
    try:
        lr = float(lr_str.replace("-", "e-").replace("+", "e+"))
    except ValueError:
        lr = lr_str
    result = {
        "lr": lr,
        "steps": int(steps),
        "eps": float(eps),
        "warmup_steps": int(warmup) if warmup is not None else 0,
        "lambda_kl": float(lambda_kl) if lambda_kl is not None else 1.0,
    }
    if align is not None:
        result["align"] = float(align)
    return result


def load_results_csv(path: Path) -> dict[str, float]:
    """Load results.csv as metric -> value dict."""
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row.get("metric", "").strip()
            val_str = row.get("value", "").strip()
            if metric and val_str:
                try:
                    out[metric] = float(val_str)
                except ValueError:
                    out[metric] = val_str
    return out


def compute_auc_from_metrics(metrics: dict[str, float], task: int, max_step: int) -> float:
    """
    Compute normalized area under curve for phase2_step{N}_task{T}_sr.
    AUC is trapezoidal, normalized by max_step so result is in [0, 1].
    """
    pattern = re.compile(rf"phase2_step(\d+)_task{task}_sr")
    points = []
    for key, val in metrics.items():
        mo = pattern.fullmatch(key)
        if mo and isinstance(val, (int, float)):
            step = int(mo.group(1))
            points.append((step, float(val)))
    if not points:
        return float("nan")
    points.sort(key=lambda p: p[0])
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    # Trapezoidal rule
    auc = 0.0
    for i in range(len(x) - 1):
        auc += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2.0
    # Normalize by max possible area (max_step * 1.0)
    if max_step <= 0:
        return float("nan")
    return auc / max_step


def harmonic_mean_success_rate(t1: float, t2: float) -> float:
    """Harmonic mean of two success rates in [0,1]."""
    a, b = float(t1), float(t2)
    s = a + b
    if s <= 0:
        return 0.0
    return 2.0 * a * b / s


def _numeric_mean(values: list) -> float | str:
    """Arithmetic mean of numeric values; skip empty strings and NaN."""
    nums: list[float] = []
    for v in values:
        if v == "" or v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(x):
            continue
        nums.append(x)
    return round(sum(nums) / len(nums), 4) if nums else ""


def _setting_key(row: dict) -> tuple:
    """Key for grouping: all hyperparameters + seed, excluding task pair and metrics."""
    return (
        row["mode"],
        row["lr"],
        row["steps"],
        row["eps"],
        row["seed"],
        row.get("align", ""),
        row["warmup_steps"],
        row["lambda_kl"],
    )


def build_general_rows(
    rows: list[dict],
    *,
    include_align: bool,
) -> list[dict]:
    """One row per setting: mean of metrics over all task pairs."""
    metric_keys = [
        "task1_success_rate",
        "task2_success_rate",
        "avg_success_rate",
        "harmonic_mean_success_rate",
        "auc_task1",
        "auc_task2",
        "avg_auc",
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[_setting_key(r)].append(r)

    general: list[dict] = []
    for key, grp in sorted(groups.items(), key=lambda x: x[0]):
        mode, lr, steps, eps, seed, align, warmup, lambda_kl = key
        out: dict = {
            "mode": mode,
            "lr": lr,
            "steps": steps,
            "eps": eps,
            "seed": seed,
            "n_pairs": len(grp),
            "warmup_steps": warmup,
            "lambda_kl": lambda_kl,
        }
        if include_align:
            out["align"] = align if align != "" else ""
        for mk in metric_keys:
            out[mk] = _numeric_mean([g.get(mk) for g in grp])
        general.append(out)
    return general


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate augment_finetune results into aggregate_results.csv"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Root folder containing task subfolders (e.g. results/augment_finetune)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: {results_dir}/aggregate_results.csv)",
    )
    parser.add_argument(
        "--output-general",
        type=Path,
        default=None,
        help="General aggregate CSV (means over task pairs; default: {results_dir}/aggregate_results_general.csv)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Only include runs with these step counts (e.g. --steps 200 or --steps 100 200). Omit to include all.",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Not a directory: {results_dir}")

    output_path = args.output or (results_dir / "aggregate_results.csv")
    output_general_path = args.output_general or (results_dir / "aggregate_results_general.csv")

    # Find all results.csv files
    results_files = list(results_dir.rglob("results.csv"))
    rows = []

    for csv_path in results_files:
        # Path relative to results_dir: task_folder / hyperparam_folder / results.csv
        rel = csv_path.relative_to(results_dir)
        parts = rel.parts
        if len(parts) < 3:
            continue
        task_folder_name = parts[0]
        hyperparam_folder_name = parts[1]

        task_info = parse_task_folder(task_folder_name)
        hyper_info = parse_hyperparam_folder(hyperparam_folder_name)
        if task_info is None or hyper_info is None:
            continue
        if args.steps is not None and hyper_info["steps"] not in args.steps:
            continue

        metrics = load_results_csv(csv_path)
        task1_sr = metrics.get("phase2_eval_task1_success_rate")
        task2_sr = metrics.get("phase2_eval_task2_success_rate")
        if task1_sr is None or task2_sr is None:
            continue

        max_step = hyper_info.get("steps", 200)
        auc_t1 = compute_auc_from_metrics(metrics, 1, max_step)
        auc_t2 = compute_auc_from_metrics(metrics, 2, max_step)
        avg_sr = round((float(task1_sr) + float(task2_sr)) / 2.0, 4)
        hmean_sr = round(harmonic_mean_success_rate(task1_sr, task2_sr), 4)

        auc1_val = round(auc_t1, 4) if not math.isnan(auc_t1) else None
        auc2_val = round(auc_t2, 4) if not math.isnan(auc_t2) else None
        avg_auc = ""
        if auc1_val is not None and auc2_val is not None:
            avg_auc = round((auc1_val + auc2_val) / 2.0, 4)

        row = {
            "task1": task_info["task1"],
            "task2": task_info["task2"],
            "seed": task_info["seed"],
            "mode": task_info["mode"],
            "lr": hyper_info["lr"],
            "steps": hyper_info["steps"],
            "eps": hyper_info["eps"],
            "task1_success_rate": round(float(task1_sr), 4),
            "task2_success_rate": round(float(task2_sr), 4),
            "avg_success_rate": avg_sr,
            "harmonic_mean_success_rate": hmean_sr,
            "auc_task1": auc1_val if auc1_val is not None else "",
            "auc_task2": auc2_val if auc2_val is not None else "",
            "avg_auc": avg_auc,
        }
        if "align" in hyper_info:
            row["align"] = hyper_info["align"]
        row["warmup_steps"] = hyper_info["warmup_steps"]
        row["lambda_kl"] = hyper_info["lambda_kl"]
        rows.append(row)

    if not rows:
        print("No valid results found.")
        return

    # Determine all columns (some runs may have align)
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = [
        "task1",
        "task2",
        "seed",
        "mode",
        "lr",
        "steps",
        "eps",
    ]
    if "align" in all_keys:
        fieldnames.append("align")
    fieldnames.extend(["warmup_steps", "lambda_kl"])
    fieldnames.extend(
        [
            "task1_success_rate",
            "task2_success_rate",
            "avg_success_rate",
            "harmonic_mean_success_rate",
            "auc_task1",
            "auc_task2",
            "avg_auc",
        ]
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output_path}")

    general_rows = build_general_rows(rows, include_align="align" in all_keys)
    gen_fieldnames = [
        "mode",
        "lr",
        "steps",
        "eps",
        "seed",
        "n_pairs",
    ]
    if "align" in all_keys:
        gen_fieldnames.append("align")
    gen_fieldnames.extend(["warmup_steps", "lambda_kl"])
    gen_fieldnames.extend(
        [
            "task1_success_rate",
            "task2_success_rate",
            "avg_success_rate",
            "harmonic_mean_success_rate",
            "auc_task1",
            "auc_task2",
            "avg_auc",
        ]
    )

    with open(output_general_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=gen_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in general_rows:
            writer.writerow(row)

    print(f"Wrote {len(general_rows)} rows to {output_general_path}")


if __name__ == "__main__":
    main()
