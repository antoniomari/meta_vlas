## Fine-tune Model on First Task from libero_90

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*shape requires ndarray.*")
warnings.filterwarnings("ignore", message=".*linear_util.wrap_init.*")

# Suppress absl logging (used by JAX)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings if any

import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('jax').setLevel(logging.ERROR)
logging.getLogger('OpenGL').setLevel(logging.ERROR)

import h5py
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple
import jax.numpy as jnp

os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'
import sys
import csv
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt

sys.path.append("./meta_libero")


from utils import run_evaluation, create_policy
import openpi.models.model as _model
from openpi.shared import image_tools
from openpi.models.model import IMAGE_RESOLUTION
from utils import train_model_on_fly
from libero_dataset import prepare_task_dataset, override_create_torch_dataset
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import config as _config
from openpi.models import model as _model
import openpi.shared.download as download



def main(task_id: int = 0, seed: int = 0):
    # Configuration
    task_suite_name = "libero_10"
    num_trials = 50
    checkpoint_path = "/cluster/home/anmari/.cache/openpi/openpi-assets/checkpoints/pi05_libero"

    # Create results directory
    results_dir = Path("meta_libero/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load the full model (not just encoders)
    print("Loading full model")
    train_config = _config.get_config("pi05_libero")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    print("Model loaded successfully!")

    # Track evaluation results: list of (step, success_rate)
    eval_results = []
    # CSV file for incremental saving
    csv_filename = results_dir / f"{task_suite_name}_{task_id}.csv"

    # Helper function to append result to CSV
    def save_eval_result(step: int, acc: float, is_first: bool = False):
        mode = 'w' if is_first else 'a'
        with open(csv_filename, mode, newline='') as f:
            writer = csv.writer(f)
            if is_first:
                writer.writerow(['train_step', 'mean_accuracy'])
            writer.writerow([step, acc])
        print(f"  -> Saved to {csv_filename.name}")

    # Evaluate policy before fine-tuning (step 0)
    print("\n" + "="*60)
    print("Evaluating BEFORE fine-tuning (step 0)")
    print("="*60)
    # Note: this will create jax.random.key(0) internally
    # if we try to pass a seed here the code will crash
    config = _config.get_config("pi05_libero")
    policy = create_policy(model, config, checkpoint_path)
    success_rate = run_evaluation(
        policy=policy,
        task_suite_name=task_suite_name,
        task_id=task_id,
        num_trials=num_trials,
        save_video=False,
        seed=seed,
    )
    eval_results.append((0, success_rate))
    save_eval_result(0, success_rate, is_first=True)

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Task: {task_suite_name} task {task_id}")
    print(f"Total training steps: {total_steps}")
    print(f"\nEvaluation Results:")
    for step, acc in eval_results:
        print(f"  Step {step:4d}: {acc*100:.1f}% success rate")
    print(f"\nResults saved to: {results_dir}")
    print(f"  - {csv_filename.name}")
    print(f"  - {plot_filename.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune model on a single LIBERO task")
    parser.add_argument("--task_id", type=int, default=0, help="Task ID to fine-tune on (default: 0)")
    args = parser.parse_args()
    main(task_id=args.task_id)
