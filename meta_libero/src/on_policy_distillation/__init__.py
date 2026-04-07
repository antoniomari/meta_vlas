"""Single-task on-policy distillation: BC filtering, PDF plots, CSV, and CLI."""

from meta_libero.src.on_policy_distillation.bc_data import (
    example_alignment_ratio,
    filter_examples_by_alignment,
    filter_examples_by_max_teacher_variance,
    strip_bc_metadata,
)
from meta_libero.src.on_policy_distillation.constants import (
    CHECKPOINT_DIR,
    FULL_EXPERIMENT_EVAL_EPISODES,
    FULL_EXPERIMENT_EVAL_INTERVAL,
    STUDENT_FINAL_EVAL_EPISODES,
    TASK_SUITE_NAME,
)
from meta_libero.src.on_policy_distillation.main import main
from meta_libero.src.on_policy_distillation.plotting import (
    plot_distillation_iter_rollout_metrics_pdf,
    plot_losses_pdf,
)

__all__ = [
    "CHECKPOINT_DIR",
    "FULL_EXPERIMENT_EVAL_EPISODES",
    "FULL_EXPERIMENT_EVAL_INTERVAL",
    "STUDENT_FINAL_EVAL_EPISODES",
    "TASK_SUITE_NAME",
    "example_alignment_ratio",
    "filter_examples_by_alignment",
    "filter_examples_by_max_teacher_variance",
    "main",
    "plot_distillation_iter_rollout_metrics_pdf",
    "plot_losses_pdf",
    "strip_bc_metadata",
]
