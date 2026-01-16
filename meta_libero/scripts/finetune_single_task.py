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
from openpi.models import model as _model
import openpi.shared.download as download
from libero_dataset import override_create_torch_dataset


"""
Can call policy.infer(element), where element is structured this way,
following the preprocessing in main.py.

    element = {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(task_description),
    }
"""


"""

episode demo_0 with 148 transitions
    key: actions with shape (148, 7)
    key: dones with shape (148,)
    key: obs
        observation key agentview_rgb with shape (148, 128, 128, 3)
        observation key ee_ori with shape (148, 3)
        observation key ee_pos with shape (148, 3)
        observation key ee_states with shape (148, 6)
        observation key eye_in_hand_rgb with shape (148, 128, 128, 3)
        observation key gripper_states with shape (148, 2)
        observation key joint_states with shape (148, 7)
    key: rewards with shape (148,)
    key: robot_states with shape (148, 9)
    key: states with shape (148, 110)


                  # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Img shapes: (128, 128, 3) TODO: check
                    # Wrist img shapes: (128, 128, 3) TODO: check

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img, # TODO: check shape (128x128x3)
                            "observation/wrist_image": wrist_img, # TODO: same 128x128x3
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"], # TODO: check shape (3,)
                                    _quat2axisangle(obs["robot0_eef_quat"]), # TODO: check shape (3,)
                                    obs["robot0_gripper_qpos"], # TODO: check shape (1,)
                                )
                            ),
                            "prompt": str(task_description),
                        }
"""

def main(task_id: int = 6):
    # Configuration
    task_suite_name = "libero_10"
    num_trials = 50
    total_steps = 500
    eval_interval = 100
    seed = 0  # Global seed for reproducibility
    checkpoint_path = "/cluster/home/anmari/.cache/openpi/openpi-assets/checkpoints/pi05_libero"

    # Create results directory
    results_dir = Path("meta_libero/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # {"task_index": 6, "task": "put both moka pots on the stove"}
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


    with override_create_torch_dataset("example", task_index=task_id, load_in_memory=True):
        data_loader = _data_loader.create_data_loader(
            config,
            sharding=None,
            shuffle=True,
        )

    dataset = data_loader._data_loader._data_loader.dataset
    training_iterator = iter(data_loader)

    # Count batches for info
    print(f"Total samples: {len(dataset)}\n")

    # Fine-tune the model
    print("\nStarting fine-tuning...")
    print("Training hyperparameters:")
    print(f"  Learning rate: 2.5e-5")
    print(f"  Total steps: {total_steps}")
    print(f"  Eval interval: {eval_interval}")
    print(f"  Batch size: 64")
    print(f"  Warmup steps: 50")

    # First 100 steps
    trained_model, train_losses, train_state = train_model_on_fly(
        model=model,
        training_data_loader=training_data_loader,
        learning_rate=2.5e-5,
        num_steps=100,
        warmup_steps=50,
        weight_decay=0.0,
        log_interval=50,
        seed=seed,
    )

    # Evaluate after first 100 steps
    print("\n" + "="*60)
    print(f"Evaluating after step {eval_interval}")
    print("="*60)
    policy = create_policy(trained_model, _config.get_config("pi05_libero"), checkpoint_path)
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
            training_data_loader=training_data_loader,
            num_steps=100,
            log_interval=50,
            resume_train_state=train_state,
            resume_losses=train_losses,
            seed=seed,
        )

        # Evaluate
        print("\n" + "="*60)
        print(f"Evaluating after step {current_step}")
        print("="*60)
        policy = create_policy(trained_model, _config.get_config("pi05_libero"), checkpoint_path)
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
