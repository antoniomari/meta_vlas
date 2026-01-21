## Fine-tune Model on First Task from libero_90

# CRITICAL: Suppress warnings BEFORE any other imports (even os!)
import sys
import logging
import warnings
warnings.filterwarnings("ignore")  # Suppress ALL warnings
# Specifically suppress the JAX shape deprecation warning from Flax
warnings.filterwarnings("ignore", message=".*shape requires ndarray or scalar arguments.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax.core.scope")

# Now set environment variables
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings if any
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'  # Cleaner error messages

import dataclasses

class VersionWarningFilter(logging.Filter):
    def filter(self, record):
        # avoid lerobot warning
        return "is in 2.0 format" not in record.getMessage()
logging.getLogger().addFilter(VersionWarningFilter())



import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('jax').setLevel(logging.ERROR)
logging.getLogger('OpenGL').setLevel(logging.ERROR)

import h5py
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple
import jax.numpy as jnp
import jax
import sys
import csv
import argparse
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt

sys.path.append("./meta_libero")

# To debug jax errors
# jax.config.update('jax_disable_jit', True)


from utils import run_evaluation, create_policy
from openpi.shared import image_tools
from openpi.models.model import IMAGE_RESOLUTION
from utils import train_model_on_fly
from libero_dataset import prepare_task_dataset, override_create_torch_dataset
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import weight_loaders as _weight_loaders
from openpi.models import model as _model
import openpi.shared.download as download
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from libero_dataset import override_create_torch_dataset
import flax.nnx as nnx
import flax.traverse_util as traverse_util



import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def main(task_id: int = 8, lr: float = 2.5e-5,
use_base_model: bool = False, batch_size: int = 64):
    # Configuration
    task_suite_name = "libero_10"
    num_trials = 50
    total_steps = 200 # was 1000
    eval_interval = 20 # was 100
    seed = 0  # Global seed for reproducibility

    # IMPORTANT: checkpoint_path should ALWAYS point to pi05_libero for assets/norm stats
    # This is used by create_policy for evaluation - it needs Libero-specific assets
    checkpoint_path = "/cluster/home/anmari/.cache/openpi/openpi-assets/checkpoints/pi05_libero"

    # Choose which weights to load based on use_base_model flag
    # NOTE: We ALWAYS use the pi05_libero config (model architecture, data config, etc.)
    # We ONLY change which weights/checkpoint to load
    if use_base_model:
        # Load weights from base pi0.5 model (not fine-tuned on Libero)
        checkpoint_gs_path = "gs://openpi-assets/checkpoints/pi05_base"
        checkpoint_name = "pi05_base"
        print("="*60)
        print("Using BASE pi0.5 model weights (not pre-trained on Libero)")
        print("Config: pi05_libero (same model architecture and data settings)")
        print(f"Weights: {checkpoint_gs_path}")
        print(f"Assets: pi05_libero (for Libero-specific normalization)")
        print("="*60)
    else:
        # Load weights from pi0.5 model already fine-tuned on Libero
        checkpoint_gs_path = "gs://openpi-assets/checkpoints/pi05_libero"
        checkpoint_name = "pi05_libero"
        print("="*60)
        print("Using pi0.5 model weights PRE-TRAINED on Libero")
        print("Config: pi05_libero")
        print(f"Weights: {checkpoint_gs_path}")
        print(f"Assets: pi05_libero")
        print("="*60)

    # Create results directory
    model_suffix = "_base" if use_base_model else ""
    results_dir = Path(f"meta_libero/results/lr_{lr}{model_suffix}_b{batch_size}")
    results_dir.mkdir(parents=True, exist_ok=True)


    # Note: task_id for the dataset is different
    dataset_task_id = 6

    # IMPORTANT: Always use pi05_libero config for consistency
    # This ensures we have the correct:
    # - Model architecture (pi05=True, action_horizon=10, etc.)
    # - Data configuration (prompt_from_task, frame sampling, etc.)
    # - Normalization stats and assets for Libero
    config = _config.get_config("pi05_libero")

    # Download and load weights using the simple, working method
    # This properly sets up JIT compilation (unlike the complex weight_loader approach)
    print(f"Downloading checkpoint from {checkpoint_gs_path}...")
    checkpoint_dir = download.maybe_download(checkpoint_gs_path)

    print(f"Loading model with weights from {checkpoint_name}...")
    t0 = time.perf_counter()
    model = config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    t1 = time.perf_counter()

    print(f"\n✓ Model loaded successfully with weights from {checkpoint_name}")
    print(f"✓ Loading took {t1-t0:.2f} seconds")



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
    skip_first_eval = False
    if not skip_first_eval:
        print("\n" + "="*60)
        print(f"Evaluating before fine-tuning")
        print("="*60)

        # Note: change to True to save videos
        save_videos = False
        video_out_path = results_dir / f"videos"
        video_out_path.mkdir(parents=True, exist_ok=True)

        policy = create_policy(model, config, checkpoint_path)
        success_rate = run_evaluation(
            policy=policy,
            task_suite_name=task_suite_name,
            task_id=task_id,
            num_trials=num_trials,
            save_video=save_videos,
            seed=seed,
        )
        eval_results.append((0, success_rate))
        save_eval_result(0, success_rate, is_first=True)


    with override_create_torch_dataset("example", task_index=dataset_task_id):
        # Batch size configuration
        config = dataclasses.replace(
            config,
            batch_size=batch_size # 64 # 32,  # Normal training with JIT (use 4 if profiling without JIT)
        )
        data_loader = _data_loader.create_data_loader(
            config,
            sharding=None,
            shuffle=True,
        )

    dataset = data_loader._data_loader._data_loader.dataset
    # training_iterator = iter(data_loader)

    # Count batches for info
    print(f"Total samples: {len(dataset)}\n")

    # Fine-tune the model
    print("\nStarting fine-tuning...")
    print("Training hyperparameters:")
    print(f"  Learning rate: 2.5e-5")
    print(f"  Total steps: {total_steps}")
    print(f"  Eval interval: {eval_interval}")
    print(f"  Batch size: {batch_size}")
    print(f"  Warmup steps: 50")


    # First 100 steps
    trained_model, train_losses, train_state = train_model_on_fly(
        model=model,
        training_data_loader=data_loader,
        config=config,
        learning_rate=lr,
        num_steps=eval_interval,
        warmup_steps=1000,
        weight_decay=0.0,
        log_interval=50,
        seed=seed,
    )

    # Evaluate after first 100 steps
    print("\n" + "="*60)
    print(f"Evaluating after step {eval_interval}")
    print("="*60)
    policy = create_policy(trained_model, config, checkpoint_path)
    success_rate = run_evaluation(
        policy=policy,
        task_suite_name=task_suite_name,
        task_id=task_id,
        num_trials=num_trials,
        save_video=False,
        seed=seed,
    )
    eval_results.append((eval_interval, success_rate))
    save_eval_result(eval_interval, success_rate)

    # Continue fine-tuning for remaining steps (400 more steps in 4 chunks of 100)
    num_remaining_chunks = (total_steps - eval_interval) // eval_interval
    for i in range(num_remaining_chunks):
        current_step = (i + 2) * eval_interval  # 200, 300, 400, 500

        print("\n" + "="*60)
        print(f"Training steps {current_step - eval_interval} to {current_step}")
        print("="*60)

        # Continue fine-tuning (optimizer state preserved from resume_train_state)
        trained_model, train_losses, train_state = train_model_on_fly(
            model=trained_model,  # ignored when resuming
            training_data_loader=data_loader,
            config=config,
            num_steps=eval_interval,
            log_interval=50,
            resume_train_state=train_state,
            resume_losses=train_losses,
            seed=seed,
        )

        # Evaluate
        print("\n" + "="*60)
        print(f"Evaluating after step {current_step}")
        print("="*60)
        policy = create_policy(trained_model, config, checkpoint_path)
        success_rate = run_evaluation(
            policy=policy,
            task_suite_name=task_suite_name,
            task_id=task_id,
            num_trials=num_trials,
            save_video=False,
            seed=seed,
        )
        eval_results.append((current_step, success_rate))
        save_eval_result(current_step, success_rate)

    # CSV already saved incrementally
    print(f"\nEvaluation results saved incrementally to {csv_filename}")

    # Plot and save losses
    plot_filename = results_dir / f"{task_suite_name}_{task_id}_losses.pdf"
    print(f"\nSaving losses plot to {plot_filename}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, linewidth=0.5, alpha=0.7)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - {task_suite_name} Task {task_id}')
    plt.grid(True, alpha=0.3)

    # Add smoothed line
    window_size = min(50, len(train_losses) // 10) if len(train_losses) > 10 else 1
    if window_size > 1:
        smoothed = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(train_losses)), smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
        plt.legend()

    plt.tight_layout()
    plt.savefig(plot_filename, format='pdf', dpi=150)
    plt.close()
    print(f"Saved losses plot")

    # Plot and save evaluation accuracy vs gradient steps
    acc_plot_filename = results_dir / f"{task_suite_name}_{task_id}_accuracy.pdf"
    print(f"\nSaving accuracy plot to {acc_plot_filename}")

    plt.figure(figsize=(10, 6))

    # Extract steps and accuracies from eval_results
    if eval_results:
        steps, accuracies = zip(*eval_results)
        accuracies_percent = [acc * 100 for acc in accuracies]  # Convert to percentage

        # Plot with markers and line
        plt.plot(steps, accuracies_percent, marker='o', linewidth=2, markersize=8,
                 label=f'LR={lr}', color='#1f77b4')

        # Add shaded confidence region (optional, just for visual appeal)
        plt.fill_between(steps,
                         [max(0, a-5) for a in accuracies_percent],
                         [min(100, a+5) for a in accuracies_percent],
                         alpha=0.2, color='#1f77b4')

    plt.xlabel('# Gradient Steps', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title(f'Learning rate = {lr}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 105)  # Set y-axis range from 0 to 100%
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(acc_plot_filename, format='pdf', dpi=150)
    plt.close()
    print(f"Saved accuracy plot")

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
    print(f"  - {acc_plot_filename.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune model on a single LIBERO task")
    parser.add_argument("--task_id", type=int, default=8, help="Task ID to fine-tune on (default: 8)")
    parser.add_argument("--lr", type=float, default=2.5e-5, help="Learning rate (default: 2.5e-4)")
    parser.add_argument("--base-model", action="store_true", help="Use base pi0.5 model instead of Libero pre-trained model")
    parser.add_argument("--b", type=int, default=64, help="Batch size (default: 64)")
    args = parser.parse_args()
    main(task_id=args.task_id, lr=args.lr, use_base_model=args.base_model, batch_size=args.b)
