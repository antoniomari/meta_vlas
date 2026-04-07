"""Periodic student-only eval during full-experiment mode."""

from pathlib import Path

from meta_libero.src.on_policy_distillation.constants import (
    CHECKPOINT_DIR,
    FULL_EXPERIMENT_EVAL_EPISODES,
    TASK_SUITE_NAME,
)
from meta_libero.src.ttt import create_policy, run_evaluation


def periodic_student_eval_10ep(
    *,
    student_model,
    config,
    task_id: int,
    seed: int,
    outer_iter: int,
    video_dir: Path,
    save_video: bool,
) -> float:
    """10-episode student eval without teacher (for --full_experiment periodic curve)."""
    out = video_dir / "periodic_eval_10ep" / f"iter_{outer_iter:03d}"
    out.mkdir(parents=True, exist_ok=True)
    pol = create_policy(
        student_model,
        config,
        CHECKPOINT_DIR,
        rng_seed=seed + outer_iter * 100_003 + 444_444,
    )
    sr, _ = run_evaluation(
        policy=pol,
        train_config=config,
        num_trials=FULL_EXPERIMENT_EVAL_EPISODES,
        task_suite_name=TASK_SUITE_NAME,
        task_id=task_id,
        save_video=save_video,
        video_out_path=str(out),
        seed=seed + outer_iter + 888_888,
        teacher_policy=None,
        show_progress_bar=True,
    )
    return float(sr)
