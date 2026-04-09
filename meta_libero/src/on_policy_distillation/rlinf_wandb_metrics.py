"""RLinf / embodied_runner-style metric keys for W&B parity (undefined → 0)."""

from __future__ import annotations

from typing import Any

import numpy as np

_ROLLOUT_KEYS = (
    "rollout/env_info/task_1_success",
    "rollout/rewards",
    "rollout/returns_min",
    "rollout/returns_mean",
    "rollout/returns_max",
    "rollout/env_info/task_id",
    "rollout/env_info/task_0_success",
    "rollout/env_info/reward",
    "rollout/env_info/return",
    "rollout/env_info/episode_len",
    "rollout/advantages_min",
    "rollout/advantages_mean",
    "rollout/advantages_max",
)


def rlinf_style_train_metrics(
    *,
    loss: float,
    lr: float,
    step_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Map BC / diffusion training to embodied `train/*` scalars (PPO-specific → 0)."""
    step_info = step_info or {}
    grad = float(step_info.get("grad_norm") or 0.0)
    lv = float(loss)
    return {
        "train/total/loss": lv,
        "train/rl/loss": 0.0,
        "train/actor/raw_loss": lv,
        "train/actor/policy_clipfrac": 0.0,
        "train/actor/ppo_kl": 0.0,
        "train/actor/policy_loss": lv,
        "train/actor/lr": float(lr),
        "train/actor/grad_norm": grad,
        "train/actor/entropy_loss": 0.0,
    }


def _collect_grpo_rewards_and_advantages(
    episode_metrics: list[dict[str, Any]] | None,
) -> tuple[list[float], list[float]]:
    rewards: list[float] = []
    advantages: list[float] = []
    if not episode_metrics:
        return rewards, advantages
    for ep in episode_metrics:
        for row in ep.get("group_sampling_trace") or []:
            for d in row.get("dists") or []:
                rewards.append(-float(d))
            for a in row.get("grpo_advantages") or []:
                advantages.append(float(a))
    return rewards, advantages


def rlinf_style_rollout_metrics(
    *,
    task_id: int,
    success_rate: float,
    episode_metrics: list[dict[str, Any]] | None,
) -> dict[str, float]:
    """Aggregate distillation rollouts into embodied `rollout/*` keys."""
    out: dict[str, float] = {k: 0.0 for k in _ROLLOUT_KEYS}
    tid = int(task_id)
    sr = float(success_rate)
    out["rollout/env_info/task_id"] = float(tid)
    out["rollout/env_info/task_0_success"] = sr if tid == 0 else 0.0
    out["rollout/env_info/task_1_success"] = sr if tid == 1 else 0.0

    lens: list[float] = []
    ep_success: list[float] = []
    if episode_metrics:
        for ep in episode_metrics:
            ns = ep.get("num_steps")
            if ns is not None:
                lens.append(float(ns))
            ep_success.append(1.0 if ep.get("success") else 0.0)

    if lens:
        out["rollout/env_info/episode_len"] = float(np.mean(lens))

    rewards, advantages = _collect_grpo_rewards_and_advantages(episode_metrics)

    if rewards:
        rarr = np.asarray(rewards, dtype=np.float64)
        out["rollout/rewards"] = float(np.mean(rarr))
        out["rollout/returns_min"] = float(np.min(rarr))
        out["rollout/returns_mean"] = float(np.mean(rarr))
        out["rollout/returns_max"] = float(np.max(rarr))
    elif ep_success:
        arr = np.asarray(ep_success, dtype=np.float64)
        out["rollout/rewards"] = float(np.mean(arr))
        out["rollout/returns_min"] = float(np.min(arr))
        out["rollout/returns_mean"] = float(np.mean(arr))
        out["rollout/returns_max"] = float(np.max(arr))
    else:
        out["rollout/rewards"] = sr
        out["rollout/returns_min"] = sr
        out["rollout/returns_mean"] = sr
        out["rollout/returns_max"] = sr

    if advantages:
        aarr = np.asarray(advantages, dtype=np.float64)
        out["rollout/advantages_min"] = float(np.min(aarr))
        out["rollout/advantages_mean"] = float(np.mean(aarr))
        out["rollout/advantages_max"] = float(np.max(aarr))

    out["rollout/env_info/reward"] = out["rollout/rewards"]
    out["rollout/env_info/return"] = out["rollout/returns_mean"]

    return out
