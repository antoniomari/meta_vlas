"""Run directory naming for single-task on-policy distillation."""

from pathlib import Path

from meta_libero.src.on_policy_distillation.paths import results_root


def ensure_run_dir(
    task_id: int,
    seed: int,
    teacher_lr: float,
    teacher_steps: int,
    bc_lr: float,
    bc_steps: int,
    max_iters: int,
    single_episode: bool = False,
    lora: bool = False,
    action_expert_only: bool = False,
    alignment_ratio_threshold: float | None = None,
    align_min: float | None = None,
    rollout_episodes: int = 1,
    rollout_num_envs: int = 1,
    full_experiment: bool = False,
    student_action_merge: float = 0.0,
    group_size: int = 1,
    teacher_group_size: int = 1,
    temporal_decay: float = 1.0,
    l1_bc_loss: bool = False,
    kl_lambda: float = 0.0,
    max_teacher_variance: float | None = None,
    student_pretraining_steps: int = 0,
    grpo_like: bool = False,
    grpo_trust_eps: float | None = None,
    grpo_weight: str | None = None,
    grpo_weight_eps: float = 1e-8,
    distill_collect_every: int = 1,
    grpo_n_epoch: int | None = None,
) -> tuple[Path, Path, Path]:
    """Return (run_dir, video_dir, losses_pdf)."""
    folder_base = "on_policy_distillation_single" if single_episode else "on_policy_distillation"
    if lora:
        folder_base = f"{folder_base}_lora"
    elif action_expert_only:
        folder_base = f"{folder_base}_action_expert"
    parent = results_root() / folder_base / f"task{task_id}_seed{seed}"
    base_name = (
        f"teacher_lr{teacher_lr}_steps{teacher_steps}_bc_lr{bc_lr}_steps{bc_steps}_maxiters{max_iters}"
    )
    if rollout_episodes != 1:
        base_name = f"{base_name}_rolloutEp{rollout_episodes}"
    if int(rollout_num_envs) > 1:
        base_name = f"{base_name}_nenv{int(rollout_num_envs)}"
    if alignment_ratio_threshold is not None:
        base_name = f"{base_name}_align{alignment_ratio_threshold}"
    elif align_min is not None:
        base_name = f"{base_name}_alignmin{align_min}"
    if full_experiment:
        base_name = f"{base_name}_full"
    if abs(float(student_action_merge)) > 1e-12:
        base_name = f"{base_name}_sam{student_action_merge:g}"
    if int(group_size) != 1:
        base_name = f"{base_name}_g{int(group_size)}"
    if int(teacher_group_size) != 1:
        base_name = f"{base_name}_tg{int(teacher_group_size)}"
    if abs(float(temporal_decay) - 1.0) > 1e-12:
        base_name = f"{base_name}_td{temporal_decay:g}"
    if l1_bc_loss:
        base_name = f"{base_name}_l1bc"
    if abs(float(kl_lambda)) > 1e-12:
        base_name = f"{base_name}_kl{kl_lambda:g}"
    if max_teacher_variance is not None:
        base_name = f"{base_name}_mtv{max_teacher_variance:g}"
    if int(student_pretraining_steps) > 0:
        base_name = f"{base_name}_spts{int(student_pretraining_steps)}"
    if grpo_like:
        base_name = f"{base_name}_grpo"
    if grpo_like and grpo_n_epoch is not None and int(grpo_n_epoch) != 1:
        base_name = f"{base_name}_ne{int(grpo_n_epoch)}"
    if grpo_trust_eps is not None:
        base_name = f"{base_name}_gte{float(grpo_trust_eps):g}"
    if grpo_weight == "mean_std":
        base_name = f"{base_name}_gwms"
        if abs(float(grpo_weight_eps) - 1e-8) > 1e-15:
            base_name = f"{base_name}_gwe{float(grpo_weight_eps):g}"
    if int(distill_collect_every) != 1:
        base_name = f"{base_name}_dce{int(distill_collect_every)}"
    base = parent / base_name
    base.mkdir(parents=True, exist_ok=True)
    video_dir = base / "videos"
    video_dir.mkdir(exist_ok=True)
    losses_pdf = base / "losses.pdf"
    return base, video_dir, losses_pdf
