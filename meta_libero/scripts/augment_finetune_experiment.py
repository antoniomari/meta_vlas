## Augment Fine-tune Experiment Script
#
# Compares:
#   1. Fine-tune on task_1 only (50 steps, lr 5e-5) -> evaluate on task_1
#   2. Fine-tune on task_2 with augmentation (task_1 pseudo-labels), validate on task_2
#      -> evaluate on task_1 and task_2
#
# Usage:
#   python augment_finetune_experiment.py --task1 0 --task2 1 [--num_episodes 10] [--num_steps 50] [--lr 5e-5]
#   python augment_finetune_experiment.py --task1 0 --task2 1 --no_augment  # Phase 2 without augmentation

# Re-exec with PYTHONWARNINGS to silence JAX/Flax deprecation warnings (must be set before interpreter starts)
import os
import sys
if "PYTHONWARNINGS" not in os.environ:
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

import logging
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*shape requires ndarray or scalar arguments.*",
)
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

import argparse
import csv
import dataclasses

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openpi.training import data_loader as _data_loader  # type: ignore
import openpi.models.model as _model  # type: ignore
import openpi.training.config as _config  # type: ignore

from meta_libero.src.dataset import (  # type: ignore
    LIBERO_90_TASK_IDS_PROMPTS,
    make_pseudo_label_inference_fn,
    override_create_torch_dataset,
)
from meta_libero.src.ttt import (  # type: ignore
    create_policy,
    run_evaluation,
    train_model_on_fly,
    load_pi05_libero_model,
    copy_model,
)

TASK_SUITE_NAME = "libero_90"
CHECKPOINT_DIR = os.getenv(
    "OPENPI_CHECKPOINT_DIR",
    str(Path.home() / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero"),
)


def _results_root() -> Path:
    return Path(os.getenv("META_LIBERO_RESULTS_DIR", "meta_libero/results"))


def _setup_logging() -> None:
    class VersionWarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "is in 2.0 format" not in record.getMessage()

    logging.getLogger().addFilter(VersionWarningFilter())
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logging.getLogger("OpenGL").setLevel(logging.ERROR)


def _ensure_run_dir(
    task1: int,
    task2: int,
    seed: int,
    mode: str,
    lr: float,
    num_steps: int,
    single_episode: bool = False,
    lora: bool = False,
    action_expert_only: bool = False,
) -> tuple[Path, Path, Path]:
    """Create run directory and return (run_dir, video_dir, losses_pdf_path).

    Structure: augment_finetune[_single][_lora|_action_expert]/task{}_task{}_seed{}{_mode}/lr{}_steps{}/
    mode: "self_replay" (default), "cotraining", or "no_augment"
    """
    folder_base = "augment_finetune_single" if single_episode else "augment_finetune"
    if lora:
        folder_base = f"{folder_base}_lora"
    elif action_expert_only:
        folder_base = f"{folder_base}_action_expert"
    suffix = "" if mode == "self_replay" else ("_noaugment" if mode == "no_augment" else "_cotraining")
    parent = _results_root() / folder_base / f"task{task1}_task{task2}_seed{seed}{suffix}"
    base = parent / f"lr{lr}_steps{num_steps}"
    base.mkdir(parents=True, exist_ok=True)
    video_dir = base / "videos"
    video_dir.mkdir(exist_ok=True)
    losses_pdf = base / "losses.pdf"
    return base, video_dir, losses_pdf


def _plot_training_curves_pdf(
    phase1_train_losses: list[float],
    phase2_train_losses: list[float],
    phase2_val_task1: list[dict],
    phase1_grad_norms: list[float],
    phase2_grad_norms: list[float],
    phase1_update_norms: list[float],
    phase2_update_norms: list[float],
    pdf_path: Path,
) -> None:
    """Create PDF with loss, grad norm, and param update norm in a single figure."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    phase1_len = len(phase1_train_losses)

    # Subplot 1: Loss
    ax = axes[0]
    if phase1_train_losses:
        ax.plot(range(phase1_len), phase1_train_losses, "b-", label="Phase 1 train (task_1)", linewidth=1.5)
    if phase2_train_losses:
        steps = [phase1_len + i for i in range(len(phase2_train_losses))]
        ax.plot(steps, phase2_train_losses, "c-", label="Phase 2 train (task_2)", linewidth=1.5)
    if phase2_val_task1:
        steps = [phase1_len + x["step"] for x in phase2_val_task1]
        losses = [x["loss"] for x in phase2_val_task1]
        ax.plot(steps, losses, "r-", label="Phase 2 validation (task_1)", linewidth=1.5)
    ax.set_ylabel("Loss")
    ax.set_title("Fine-tune loss curves")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 2: Gradient norm
    ax = axes[1]
    if phase1_grad_norms:
        ax.plot(range(len(phase1_grad_norms)), phase1_grad_norms, "b-", label="Phase 1 train", linewidth=1.5)
    if phase2_grad_norms:
        steps = [phase1_len + i for i in range(len(phase2_grad_norms))]
        ax.plot(steps, phase2_grad_norms, "c-", label="Phase 2 train", linewidth=1.5)
    ax.set_ylabel("Gradient norm")
    ax.set_title("Gradient norm per step")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 3: Param update norm (magnitude of parameter shift)
    ax = axes[2]
    if phase1_update_norms:
        ax.plot(range(len(phase1_update_norms)), phase1_update_norms, "b-", label="Phase 1 train", linewidth=1.5)
    if phase2_update_norms:
        steps = [phase1_len + i for i in range(len(phase2_update_norms))]
        ax.plot(steps, phase2_update_norms, "c-", label="Phase 2 train", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Param update norm")
    ax.set_title("Parameter shift magnitude (L2 norm of updates)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(pdf_path, format="pdf", dpi=150)
    plt.close(fig)


def _save_results_csv(run_dir: Path, results: dict) -> None:
    """Save experiment results to CSV."""
    csv_path = run_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["phase1_eval_task1_success_rate", results.get("phase1_eval_task1_success_rate")])
        writer.writerow(["phase2_eval_task1_success_rate", results.get("phase2_eval_task1_success_rate")])
        writer.writerow(["phase2_eval_task2_success_rate", results.get("phase2_eval_task2_success_rate")])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment fine-tune experiment: compare standard vs augmented fine-tuning on two tasks"
    )
    parser.add_argument("--task1", type=int, required=True, help="Task ID for phase 1 (train only)")
    parser.add_argument("--task2", type=int, required=True, help="Task ID for phase 2 (train + augment with task_1 pseudo-labels)")
    parser.add_argument("--num_episodes", type=int, default=10, help="Evaluation episodes per task")
    parser.add_argument("--num_steps", type=int, default=50, help="Fine-tuning steps per phase")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_video", action="store_true", default=True, help="Save rollout videos")
    parser.add_argument("--no_save_video", action="store_false", dest="save_video")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["self_replay", "cotraining", "no_augment"],
        default="self_replay",
        help="Phase 2 mode: self_replay (task_1 pseudo-labels), cotraining (task_2+task_1 independent sampling), no_augment (task_2 only)",
    )
    parser.add_argument(
        "--single_episode",
        action="store_true",
        help="Use single-episode datasets; results go to augment_finetune_single/",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA for fine-tuning; results go to augment_finetune[_single]_lora/",
    )
    parser.add_argument(
        "--action_expert_only",
        action="store_true",
        help="Fine-tune action expert only; results go to augment_finetune[_single]_action_expert/",
    )
    args = parser.parse_args()

    if args.lora and args.action_expert_only:
        parser.error("--lora and --action_expert_only are mutually exclusive")

    _setup_logging()

    task1, task2 = args.task1, args.task2
    num_episodes = args.num_episodes
    num_steps = args.num_steps
    lr = args.lr
    batch_size = args.batch_size
    seed = args.seed
    save_video = args.save_video

    single_episode = args.single_episode
    lora = args.lora
    action_expert_only = args.action_expert_only
    mode = args.mode
    run_dir, video_dir, losses_pdf = _ensure_run_dir(
        task1, task2, seed, mode, lr, num_steps,
        single_episode=single_episode,
        lora=lora,
        action_expert_only=action_expert_only,
    )

    print("=" * 70)
    print("Augment Fine-tune Experiment")
    print("=" * 70)
    print(f"Task 1 (train/validate): {task1}")
    print(f"Task 2 (generalization): {task2}")
    print(f"Phase 2 mode: {mode}")
    print(f"Single episode: {single_episode}")
    print(f"LoRA: {lora}")
    print(f"Action expert only: {action_expert_only}")
    print(f"Num episodes: {num_episodes}, Steps: {num_steps}, LR: {lr}")
    print(f"Results: {run_dir}")
    print("=" * 70)

    # Load base model
    print("\nLoading model...")
    model, config = load_pi05_libero_model(
        use_base_model=False,
        use_lora=lora,
        action_expert_only=action_expert_only,
    )
    config = dataclasses.replace(config, batch_size=batch_size)
    repo_id = "antoniomari/libero_90"

    results = {
        "phase1_train_losses": [],
        "phase1_train_grad_norms": [],
        "phase1_train_update_norms": [],
        "phase1_eval_task1_success_rate": None,
        "phase2_train_losses": [],
        "phase2_train_grad_norms": [],
        "phase2_train_update_norms": [],
        "phase2_val_task1_losses": [],
        "phase2_eval_task1_success_rate": None,
        "phase2_eval_task2_success_rate": None,
    }

    # -------------------------------------------------------------------------
    # Phase 1: Fine-tune on task_1 only (no augmentation)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Phase 1: Fine-tune on task_1 (no augmentation)")
    print("=" * 50)

    with override_create_torch_dataset(
        repo_id=repo_id,
        task_id=task1,
        mirror_data=True,
        single_episode=single_episode,
        augment=False,
    ):
        train_loader = _data_loader.create_data_loader(
            config, sharding=None, shuffle=True,
        )

    model_phase1 = copy_model(model, config)
    phase1_losses: list[float] = []
    phase1_grad_norms: list[float] = []
    phase1_update_norms: list[float] = []

    def _on_step_phase1(step: int, loss_val: float) -> None:
        phase1_losses.append(loss_val)

    def _on_step_info_phase1(step: int, info: dict) -> None:
        phase1_grad_norms.append(info["grad_norm"])
        phase1_update_norms.append(info["update_norm"])

    model_phase1, _, _ = train_model_on_fly(
        model=model_phase1,
        training_data_loader=train_loader,
        config=config,
        learning_rate=lr,
        num_steps=num_steps,
        warmup_steps=0,
        weight_decay=0.0,
        log_interval=max(1, num_steps // 10),
        seed=seed,
        show_progress_bar=True,
        donate_buffers=False,
        on_step_callback=_on_step_phase1,
        on_step_info_callback=_on_step_info_phase1,
    )
    model_phase1.eval()
    results["phase1_train_losses"] = phase1_losses
    results["phase1_train_grad_norms"] = phase1_grad_norms
    results["phase1_train_update_norms"] = phase1_update_norms
    print(f"Phase 1 train loss (final): {phase1_losses[-1]:.4f}" if phase1_losses else "N/A")

    # Evaluate on task_1
    policy_phase1 = create_policy(model_phase1, config, CHECKPOINT_DIR, rng_seed=seed)
    video_path_phase1 = str(video_dir / "phase1_task1")
    sr1, _ = run_evaluation(
        policy=policy_phase1,
        train_config=config,
        num_trials=num_episodes,
        task_suite_name=TASK_SUITE_NAME,
        task_id=task1,
        save_video=save_video,
        video_out_path=video_path_phase1,
        seed=seed,
    )
    results["phase1_eval_task1_success_rate"] = sr1
    print(f"Phase 1 eval task_1 success rate: {sr1 * 100:.1f}%")

    del train_loader

    # -------------------------------------------------------------------------
    # Phase 2: Fine-tune on task_2 (mode: self_replay, cotraining, or no_augment)
    # -------------------------------------------------------------------------
    phase2_augment = mode == "self_replay"
    phase2_cotraining = mode == "cotraining"
    if phase2_cotraining:
        print("\n" + "=" * 50)
        print("Phase 2: Cotraining on task_2 + task_1 (independent sampling)")
        print("=" * 50)
    elif phase2_augment:
        print("\n" + "=" * 50)
        print("Phase 2: Self-replay on task_2 with task_1 pseudo-labels")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("Phase 2: Fine-tune on task_2 only (no augmentation)")
        print("=" * 50)

    augment_inference_fn = (
        make_pseudo_label_inference_fn(policy_phase1) if phase2_augment else None
    )
    if phase2_augment:
        augment_prompt = LIBERO_90_TASK_IDS_PROMPTS.get(task1, "do nothing")
        print(f"Augmentation prompt (task_1={task1}): {augment_prompt!r}")

    with override_create_torch_dataset(
        repo_id=repo_id,
        task_id=task2,
        mirror_data=True,
        single_episode=single_episode,
        augment=phase2_augment,
        augment_task_id=task1 if phase2_augment else None,
        augment_inference_fn=augment_inference_fn,
        cotraining=phase2_cotraining,
        cotraining_task_id=task1 if phase2_cotraining else None,
    ):
        train_loader = _data_loader.create_data_loader(
            config, sharding=None, shuffle=True,
        )

    # Validation on task_1
    val_config = dataclasses.replace(config, num_workers=0)
    with override_create_torch_dataset(
        repo_id=repo_id,
        task_id=task1,
        mirror_data=True,
        single_episode=True,
        augment=False,
    ):
        val_loader_task1 = _data_loader.create_data_loader(
            val_config,
            sharding=None,
            shuffle=False,
            single_epoch=True,
        )

    phase2_train_losses: list[float] = []
    phase2_grad_norms: list[float] = []
    phase2_update_norms: list[float] = []
    phase2_val_task1: list[dict] = []

    def _on_step_phase2(step: int, loss_val: float) -> None:
        phase2_train_losses.append(loss_val)

    def _on_step_info_phase2(step: int, info: dict) -> None:
        phase2_grad_norms.append(info["grad_norm"])
        phase2_update_norms.append(info["update_norm"])

    def _on_val_task1(step: int, loss_val: float) -> None:
        phase2_val_task1.append({"step": step, "loss": loss_val})

    model_phase2 = copy_model(model_phase1, config)
    model_phase2, _, _ = train_model_on_fly(
        model=model_phase2,
        training_data_loader=train_loader,
        config=config,
        learning_rate=lr,
        num_steps=num_steps,
        warmup_steps=0,
        weight_decay=0.0,
        log_interval=max(1, num_steps // 10),
        seed=seed,
        show_progress_bar=True,
        donate_buffers=False,
        on_step_callback=_on_step_phase2,
        on_step_info_callback=_on_step_info_phase2,
        validation_data_loader=val_loader_task1,
        validation_interval=5,
        on_validation_callback=_on_val_task1,
    )
    model_phase2.eval()

    del policy_phase1, model_phase1

    results["phase2_train_losses"] = phase2_train_losses
    results["phase2_train_grad_norms"] = phase2_grad_norms
    results["phase2_train_update_norms"] = phase2_update_norms
    results["phase2_val_task1_losses"] = phase2_val_task1

    # -------------------------------------------------------------------------
    # Plot losses (before evals)
    # -------------------------------------------------------------------------
    _plot_training_curves_pdf(
        phase1_train_losses=results["phase1_train_losses"],
        phase2_train_losses=results["phase2_train_losses"],
        phase2_val_task1=results["phase2_val_task1_losses"],
        phase1_grad_norms=results["phase1_train_grad_norms"],
        phase2_grad_norms=results["phase2_train_grad_norms"],
        phase1_update_norms=results["phase1_train_update_norms"],
        phase2_update_norms=results["phase2_train_update_norms"],
        pdf_path=losses_pdf,
    )
    print(f"\nSaved losses PDF to {losses_pdf}")

    # -------------------------------------------------------------------------
    # Evaluate on task_1 and task_2
    # -------------------------------------------------------------------------
    policy_phase2 = create_policy(model_phase2, config, CHECKPOINT_DIR, rng_seed=seed)

    video_path_phase2_t1 = str(video_dir / "phase2_task1")
    sr2_t1, _ = run_evaluation(
        policy=policy_phase2,
        train_config=config,
        num_trials=num_episodes,
        task_suite_name=TASK_SUITE_NAME,
        task_id=task1,
        save_video=save_video,
        video_out_path=video_path_phase2_t1,
        seed=seed,
    )
    results["phase2_eval_task1_success_rate"] = sr2_t1
    print(f"Phase 2 eval task_1 success rate: {sr2_t1 * 100:.1f}%")

    video_path_phase2_t2 = str(video_dir / "phase2_task2")
    sr2_t2, _ = run_evaluation(
        policy=policy_phase2,
        train_config=config,
        num_trials=num_episodes,
        task_suite_name=TASK_SUITE_NAME,
        task_id=task2,
        save_video=save_video,
        video_out_path=video_path_phase2_t2,
        seed=seed,
    )
    results["phase2_eval_task2_success_rate"] = sr2_t2
    print(f"Phase 2 eval task_2 success rate: {sr2_t2 * 100:.1f}%")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    _save_results_csv(run_dir, results)
    print(f"Saved results CSV to {run_dir / 'results.csv'}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Phase 1 eval task_1: {results['phase1_eval_task1_success_rate'] * 100:.1f}%")
    print(f"Phase 2 eval task_1: {results['phase2_eval_task1_success_rate'] * 100:.1f}%")
    print(f"Phase 2 eval task_2: {results['phase2_eval_task2_success_rate'] * 100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
