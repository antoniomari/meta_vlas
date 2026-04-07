## On-policy distillation: teacher fine-tuned on task1, student trains from base on teacher-labeled rollouts.
#
# Implementation: ``meta_libero.src.on_policy_distillation`` (modular package). This file only sets up
# the environment and delegates to :func:`meta_libero.src.on_policy_distillation.main.main`.
#
# Weights & Biases: pass --wandb_project YOUR_PROJECT to log. Set credentials once via ``wandb login``
# (stores key in ~/.netrc) or set environment variable WANDB_API_KEY.
#
# Usage:
#   python on_policy_distillation.py --task 6 --teacher_steps 50 --bc_steps 20 --lr 1e-4

import os
import sys

if "PYTHONWARNINGS" not in os.environ:
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*shape requires ndarray or scalar arguments.*",
)
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*linear_util.wrap_init.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax")

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "meta_libero"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("HF_LEROBOT_HOME", str(Path(os.environ["HF_HOME"]) / "lerobot"))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import matplotlib

matplotlib.use("Agg")

from meta_libero.src.on_policy_distillation.main import main

if __name__ == "__main__":
    main()
