"""Weights & Biases helpers for on-policy distillation."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import wandb


def summarize_rollout_metrics(
    episode_metrics: list[dict[str, Any]] | None,
) -> dict[str, float]:
    """Scalar summaries for wandb from distillation_rollout_metrics-style episode dicts."""
    if not episode_metrics:
        return {}
    out: dict[str, float] = {}
    ta: list[float] = []
    tl2: list[float] = []
    cvar: list[float] = []
    tvar: list[float] = []
    rewards: list[float] = []
    gw_ms: list[float] = []
    adv_abs: list[float] = []
    for ep in episode_metrics:
        for _t, v in ep.get("teacher_alignment_ratio_by_step") or []:
            ta.append(float(v))
        for _t, v in ep.get("teacher_l2_by_env_step") or []:
            tl2.append(float(v))
        for row in ep.get("group_sampling_trace") or []:
            cvar.append(float(row.get("chunk_var_mean", 0.0)))
            for d in row.get("dists") or []:
                rewards.append(-float(d))
            wms = row.get("grpo_mean_std_weight")
            if wms is not None and np.isfinite(float(wms)):
                gw_ms.append(float(wms))
            for a in row.get("grpo_advantages") or []:
                adv_abs.append(abs(float(a)))
        for row in ep.get("teacher_group_sampling_trace") or []:
            tvar.append(float(row.get("chunk_var_mean", 0.0)))
    if ta:
        out["distill/rollout_mean_teacher_align"] = float(np.mean(ta))
    if tl2:
        out["distill/rollout_mean_teacher_student_l2"] = float(np.mean(tl2))
    if cvar:
        out["distill/rollout_mean_student_chunk_var"] = float(np.mean(cvar))
    if tvar:
        out["distill/rollout_mean_teacher_chunk_var"] = float(np.mean(tvar))
    if rewards:
        out["distill/rollout_mean_grpo_reward"] = float(np.mean(rewards))
    if gw_ms:
        out["distill/rollout_mean_grpo_mean_std_weight"] = float(np.mean(gw_ms))
    if adv_abs:
        out["distill/rollout_mean_abs_grpo_advantage"] = float(np.mean(adv_abs))
    return out


def init_wandb(
    args: argparse.Namespace,
    run_dir: Path,
    *,
    config: dict[str, Any],
) -> Any | None:
    """Initialize a Weights & Biases run; return the run object or None if disabled."""
    project = getattr(args, "wandb_project", None)
    if not project:
        return None
    tags = []
    wt = getattr(args, "wandb_tags", None)
    if wt:
        tags = [t.strip() for t in str(wt).split(",") if t.strip()]
    run = wandb.init(
        project=str(project),
        entity=(getattr(args, "wandb_entity", None) or None),
        name=(getattr(args, "wandb_run_name", None) or run_dir.name),
        group=(getattr(args, "wandb_group", None) or None),
        tags=tags or None,
        config=config,
        dir=str(run_dir),
    )
    try:
        wandb.define_metric("teacher_step")
        wandb.define_metric("teacher/*", step_metric="teacher_step")
        wandb.define_metric("sft_step")
        wandb.define_metric("sft/*", step_metric="sft_step")
        wandb.define_metric("bc_step")
        wandb.define_metric("bc/*", step_metric="bc_step")
        wandb.define_metric("outer_iter")
        wandb.define_metric("distill/*", step_metric="outer_iter")
    except Exception:
        pass
    return run
