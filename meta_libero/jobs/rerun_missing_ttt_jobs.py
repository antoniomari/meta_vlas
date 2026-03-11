#!/usr/bin/env python3
"""Detect missing TTT reptile runs from results_summary.csv and resubmit.

Usage:
  python meta_libero/jobs/rerun_missing_ttt_jobs.py \
      --results-csv meta_libero/results/ttt_new_mirrored/dataset_libero_90_reptile/action_only/results_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
from typing import Any


DEFAULT_SEEDS = (1, 2, 3)
DEFAULT_EPS = (0.0, 0.25, 0.5, 0.75, 1.0)


HYPER_DIR_RE = re.compile(
    r"^lr(?P<lr>[^_]+)_freq(?P<freq>\d+)_steps(?P<steps>\d+)"
    r"_k(?P<k>\d+)_nn(?P<nn>\d+)"
    r"(?:_eps(?P<eps>[^_]+))?_seed(?P<seed>\d+)$"
)
TASK_DIR_RE = re.compile(r"^(?P<suite>.+)_task_(?P<task>\d+)$")


def _norm_float(v: Any) -> float:
    return round(float(v), 10)


def _parse_bool(s: Any) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y"}


def _parse_layout_from_path(results_csv: Path) -> dict[str, Any]:
    parts = results_csv.parts
    # .../results/<main_dir>/<dataset_dir>/<optional_model_dir>/results_summary.csv
    if "results" not in parts:
        raise ValueError(f"Expected 'results' in path: {results_csv}")
    i = parts.index("results")
    if len(parts) < i + 4:
        raise ValueError(f"Unexpected results csv layout: {results_csv}")

    main_dir = parts[i + 1]
    dataset_dir = parts[i + 2]
    model_dir = parts[i + 3] if parts[i + 3] != "results_summary.csv" else ""

    if dataset_dir.endswith("_reptile"):
        meta_update = "tt_reptile"
    elif dataset_dir.endswith("_continual"):
        meta_update = "continual_ttt"
    elif dataset_dir.endswith("_reset"):
        meta_update = "reset"
    else:
        raise ValueError(f"Cannot infer meta_update from dataset dir: {dataset_dir}")

    if "dataset_libero_90" in dataset_dir:
        dataset_flag = "--libero-90-dataset"
    else:
        dataset_flag = ""

    layout = {
        "main_dir": main_dir,
        "dataset_dir": dataset_dir,
        "model_dir": model_dir,
        "meta_update": meta_update,
        "dataset_flag": dataset_flag,
        "mirror_data": ("mirrored" in main_dir),
        "no_reset": ("no_reset" in main_dir),
        "use_base_model": ("base_model" in main_dir),
    }
    return layout


def _collect_k_from_run_dirs(base_dir: Path) -> dict[tuple[Any, ...], set[int]]:
    """Collect known ttt_k values from existing run directory names."""
    mapping: dict[tuple[Any, ...], set[int]] = {}
    for task_dir in base_dir.glob("*_task_*"):
        if not task_dir.is_dir():
            continue
        tm = TASK_DIR_RE.match(task_dir.name)
        if not tm:
            continue
        suite = tm.group("suite")
        task_id = int(tm.group("task"))

        for run_dir in task_dir.iterdir():
            if not run_dir.is_dir():
                continue
            rm = HYPER_DIR_RE.match(run_dir.name)
            if not rm:
                continue
            key = (
                suite,
                task_id,
                _norm_float(rm.group("lr")),
                int(rm.group("freq")),
                int(rm.group("steps")),
                int(rm.group("nn")),
            )
            mapping.setdefault(key, set()).add(int(rm.group("k")))
    return mapping


def _submit_job(
    *,
    project_root: Path,
    venv_path: Path,
    log_dir: Path,
    time_limit: str,
    mem_per_cpu: str,
    gpu: str,
    row: dict[str, str],
    ttt_k: int,
    seed: int,
    eps: float,
    layout: dict[str, Any],
    dry_run: bool,
) -> None:
    lr = float(row["lr"])
    ttt_freq = int(row["ttt_frequency"])
    ttt_steps = int(row["ttt_num_steps"])
    num_trials = int(row["num_trials"])
    task_suite = row["task_suite_name"]
    task_id = int(row["task_id"])

    use_lora = _parse_bool(row["use_lora"])
    action_only = _parse_bool(row["action_expert_only"])

    model_flag = ""
    finetune_mode = "full"
    if use_lora:
        model_flag = "--use-lora"
        finetune_mode = "lora"
    elif action_only:
        model_flag = "--action-expert-only"
        finetune_mode = "action_expert_only"

    no_reset_flag = "--no_reset" if layout["no_reset"] else ""
    base_flag = "--use-base-model" if layout["use_base_model"] else ""
    mirror_off_flag = "" if layout["mirror_data"] else "--no-mirror-data"
    mirror_mode = "mirrored" if layout["mirror_data"] else "non_mirrored"
    merging_eps_flag = f"--merging_eps {eps:g}"
    dataset_flag = layout["dataset_flag"]
    meta_update = layout["meta_update"]

    job_name = (
        f"ttt_rerun_t{task_id}_s{seed}_lr{lr:.1e}"
        f"_f{ttt_freq}_st{ttt_steps}_k{ttt_k}_{meta_update}_eps{eps:g}"
    )

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time_limit}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --gpus={gpu}
#SBATCH --output={log_dir}/ttt_%j.out
#SBATCH --error={log_dir}/ttt_%j.err

cd "{project_root}"
source "{venv_path}/bin/activate"

JOB_TMP_BASE="/tmp/${{USER}}/meta_libero_${{SLURM_JOB_ID}}"
mkdir -p "${{JOB_TMP_BASE}}/tmp" "${{JOB_TMP_BASE}}/jax_cache"
chmod 700 "${{JOB_TMP_BASE}}" "${{JOB_TMP_BASE}}/tmp" "${{JOB_TMP_BASE}}/jax_cache"
export TMPDIR="${{JOB_TMP_BASE}}/tmp"
export TMP="${{TMPDIR}}"
export TEMP="${{TMPDIR}}"
export XLA_FLAGS="${{XLA_FLAGS:-}} --xla_gpu_kernel_cache_file=${{JOB_TMP_BASE}}/jax_cache/xla_gpu_kernel_cache"

python meta_libero/scripts/ttt_evaluation.py \\
  --task_suite_name "{task_suite}" \\
  --task_id {task_id} \\
  --num_trials {num_trials} \\
  --lr {lr} \\
  --ttt_frequency {ttt_freq} \\
  --ttt_num_steps {ttt_steps} \\
  --ttt_k {ttt_k} \\
  --meta_update {meta_update} \\
  {merging_eps_flag} \\
  {no_reset_flag} \\
  --seed {seed} \\
  --max_ttt_step 1000 \\
  {model_flag} \\
  --save-video \\
  {base_flag} \\
  {dataset_flag} \\
  {mirror_off_flag}
"""

    if dry_run:
        print(
            f"[DRY RUN] Would submit: {job_name} | "
            f"finetune_mode={finetune_mode} | mirror_mode={mirror_mode}"
        )
        return

    proc = subprocess.run(
        ["sbatch"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed for {job_name}:\n{proc.stderr}")
    print(
        f"{proc.stdout.strip()} | "
        f"finetune_mode={finetune_mode} | mirror_mode={mirror_mode}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun missing TTT reptile jobs from a results_summary.csv")
    parser.add_argument("--results-csv", required=True, help="Path to results_summary.csv")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--gpu", default="pro_6000:1")
    parser.add_argument("--ttt-k", type=int, default=None, help="Fallback ttt_k if not inferable from run dirs")
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    results_csv = Path(args.results_csv).resolve()
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)

    layout = _parse_layout_from_path(results_csv)
    base_dir = results_csv.parent
    k_map = _collect_k_from_run_dirs(base_dir)

    rows: list[dict[str, str]] = []
    with results_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("lr"):
                rows.append(row)
    if not rows:
        print("No rows in results_summary.csv")
        return

    # Existing (seed, eps) by base run config
    existing: dict[tuple[Any, ...], set[tuple[int, float]]] = {}
    for r in rows:
        base_key = (
            r["task_suite_name"],
            int(r["task_id"]),
            _norm_float(r["lr"]),
            int(r["ttt_frequency"]),
            int(r["ttt_num_steps"]),
            int(r["num_neighbors_fetch"]),
            _parse_bool(r["use_lora"]),
            _parse_bool(r["action_expert_only"]),
            int(r["num_trials"]),
            int(r["batch_size"]),
        )
        eps = _norm_float(r["merging_eps"]) if r.get("merging_eps", "") != "" else None
        if eps is None:
            continue
        existing.setdefault(base_key, set()).add((int(r["seed"]), eps))

    project_root = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    log_dir = Path(os.getenv("META_LIBERO_LOG_DIR", project_root / "meta_libero" / "logs"))
    venv_path = Path(os.getenv("META_VENV_PATH", project_root / ".venv"))
    log_dir.mkdir(parents=True, exist_ok=True)

    missing_jobs = []
    for base_key, done in existing.items():
        suite, task_id, lr, freq, steps, nn, use_lora, action_only, num_trials, batch_size = base_key
        k_lookup_key = (suite, task_id, lr, freq, steps, nn)
        k_candidates = k_map.get(k_lookup_key, set())
        if len(k_candidates) == 1:
            ttt_k = next(iter(k_candidates))
        elif args.ttt_k is not None:
            ttt_k = args.ttt_k
        else:
            raise ValueError(
                f"Could not infer unique ttt_k for {k_lookup_key}. "
                f"Candidates={sorted(k_candidates)}. Pass --ttt-k."
            )

        # Build a template row from any matching CSV row.
        template = None
        for r in rows:
            rk = (
                r["task_suite_name"],
                int(r["task_id"]),
                _norm_float(r["lr"]),
                int(r["ttt_frequency"]),
                int(r["ttt_num_steps"]),
                int(r["num_neighbors_fetch"]),
                _parse_bool(r["use_lora"]),
                _parse_bool(r["action_expert_only"]),
                int(r["num_trials"]),
                int(r["batch_size"]),
            )
            if rk == base_key:
                template = r
                break
        if template is None:
            continue

        for seed in DEFAULT_SEEDS:
            for eps in DEFAULT_EPS:
                key = (seed, _norm_float(eps))
                if key in done:
                    continue
                missing_jobs.append((template, ttt_k, seed, eps))

    if not missing_jobs:
        print("No missing runs detected for seeds {1,2,3} and eps {0.0,0.25,0.5,0.75,1.0}.")
        return

    print(f"Detected {len(missing_jobs)} missing runs. Submitting...")
    for row, ttt_k, seed, eps in missing_jobs:
        _submit_job(
            project_root=project_root,
            venv_path=venv_path,
            log_dir=log_dir,
            time_limit=args.time,
            mem_per_cpu=args.mem,
            gpu=args.gpu,
            row=row,
            ttt_k=ttt_k,
            seed=seed,
            eps=eps,
            layout=layout,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
