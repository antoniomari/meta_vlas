"""Shared constants for on-policy distillation experiments."""

import os
from pathlib import Path

TASK_SUITE_NAME = "libero_90"
STUDENT_FINAL_EVAL_EPISODES = 10
FULL_EXPERIMENT_EVAL_INTERVAL = 5
FULL_EXPERIMENT_EVAL_EPISODES = 10
CHECKPOINT_DIR = os.getenv(
    "OPENPI_CHECKPOINT_DIR",
    str(Path.home() / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero"),
)
