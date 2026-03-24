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
#   python augment_finetune_experiment.py --task1 0 --task2 1 --mode two_lrs  # self-replay data, dual LR (2.5e-4 / 2.5e-5)

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
    merge_model_parameters,
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
    merging_eps: float = 1.0,
    alignment_ratio_threshold: float | None = None,
    warmup_steps: int = 0,
    lambda_kl: float = 1.0,
) -> tuple[Path, Path, Path]:
    """Create run directory and return (run_dir, video_dir, losses_pdf_path).

    Structure: augment_finetune[_single][_lora|_action_expert]/task{}_task{}_seed{}{_mode}/lr{}_steps{}_eps{}[_align{}][_warmup{}][_lambdaKL{}]/
    mode: "self_replay" (default), "two_lrs" (self-replay data + dual LR), "cotraining", "on_policy_self_replay", "self_check", or "no_augment"
    merging_eps: in subfolder name next to lr and steps
    alignment_ratio_threshold: when set, append _align{threshold} to subfolder name
    warmup_steps: when > 0, append _warmup{N} to subfolder name; 0 = default, omit from name
    lambda_kl: when mode=self_check and != 1, append _lambdaKL{value} to subfolder name
    """
    folder_base = "augment_finetune_single" if single_episode else "augment_finetune"
    if lora:
        folder_base = f"{folder_base}_lora"
    elif action_expert_only:
        folder_base = f"{folder_base}_action_expert"
    if mode == "self_replay":
        suffix = ""
    elif mode == "two_lrs":
        suffix = "_two_lrs"
    elif mode == "no_augment":
        suffix = "_noaugment"
    elif mode == "on_policy_self_replay":
        suffix = "_onpolicy_self_replay"
    elif mode == "self_check":
        suffix = "_self_check"
    else:
        suffix = "_cotraining"
    parent = _results_root() / folder_base / f"task{task1}_task{task2}_seed{seed}{suffix}"
    base_name = f"lr{lr}_steps{num_steps}_eps{merging_eps}"
    if alignment_ratio_threshold is not None:
        base_name = f"{base_name}_align{alignment_ratio_threshold}"
    if warmup_steps > 0:
        base_name = f"{base_name}_warmup{warmup_steps}"
    if mode == "self_check" and lambda_kl != 1.0:
        base_name = f"{base_name}_lambdaKL{lambda_kl}"
    base = parent / base_name
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
    phase1_grad_cosine_sims: list[float],
    phase2_grad_cosine_sims: list[float],
    phase2_alignment_avg_ratios: list[float] | None = None,
    phase2_eval_by_step: list[dict] | None = None,
    pdf_path: Path | None = None,
) -> None:
    """Create PDF with loss, grad norm, param update norm, grad cosine sim, alignment, and eval success."""
    n_subplots = 4
    if phase2_alignment_avg_ratios:
        n_subplots += 1
    if phase2_eval_by_step:
        n_subplots += 1
    fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 3 * n_subplots), sharex=True)
    if n_subplots == 1:
        axes = [axes]

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
    ax.set_ylabel("Param update norm")
    ax.set_title("Parameter shift magnitude (L2 norm of updates)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 4: Gradient cosine similarity (current vs previous gradient)
    ax = axes[3]
    if phase1_grad_cosine_sims:
        ax.plot(range(len(phase1_grad_cosine_sims)), phase1_grad_cosine_sims, "b-", label="Phase 1 train", linewidth=1.5)
    if phase2_grad_cosine_sims:
        steps = [phase1_len + i for i in range(len(phase2_grad_cosine_sims))]
        ax.plot(steps, phase2_grad_cosine_sims, "c-", label="Phase 2 train", linewidth=1.5)
    ax.set_ylabel("Grad cosine sim")
    ax.set_title("Gradient cosine similarity (current vs previous)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 5: Alignment ratio (avg over task1 samples in batch)
    ax_idx = 4
    if phase2_alignment_avg_ratios:
        ax = axes[ax_idx]
        steps = [phase1_len + i for i in range(len(phase2_alignment_avg_ratios))]
        ax.plot(steps, phase2_alignment_avg_ratios, "c-", label="Avg alignment ratio", linewidth=1.5)
        ax.set_ylabel("Ratio")
        ax.set_title("Alignment ratio (avg over task1 samples in batch)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    # Subplot 6: Eval success rate during Phase 2 (every 20 steps + final)
    if phase2_eval_by_step:
        ax = axes[ax_idx]
        steps = [phase1_len + x["step"] for x in phase2_eval_by_step]
        task1_srs = [x["task1_success_rate"] * 100 for x in phase2_eval_by_step]
        task2_srs = [x["task2_success_rate"] * 100 for x in phase2_eval_by_step]
        ax.plot(steps, task1_srs, "r-o", label="Task_1 success %", markersize=4)
        ax.plot(steps, task2_srs, "g-o", label="Task_2 success %", markersize=4)
        ax.set_ylabel("Success %")
        ax.set_title("Phase 2 eval during training (every 20 steps) and final")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    axes[ax_idx - 1].set_xlabel("Step")

    plt.tight_layout()
    assert pdf_path is not None, "pdf_path is required"
    fig.savefig(pdf_path, format="pdf", dpi=150)
    plt.close(fig)


def _save_results_csv(run_dir: Path, results: dict) -> None:
    """Save experiment results to CSV."""
    csv_path = run_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["mode", results.get("mode")])
        writer.writerow(["merging_eps", results.get("merging_eps")])
        if results.get("warmup_steps", 0) > 0:
            writer.writerow(["warmup_steps", results.get("warmup_steps")])
        if results.get("lambda_kl") is not None:
            writer.writerow(["lambda_kl", results.get("lambda_kl")])
        if results.get("lr_action_expert") is not None:
            writer.writerow(["lr_action_expert", results.get("lr_action_expert")])
            writer.writerow(["lr_base", results.get("lr_base")])
        writer.writerow(["phase1_eval_task1_success_rate", results.get("phase1_eval_task1_success_rate")])
        writer.writerow(["phase2_eval_task1_success_rate", results.get("phase2_eval_task1_success_rate")])
        writer.writerow(["phase2_eval_task2_success_rate", results.get("phase2_eval_task2_success_rate")])
        for ev in results.get("phase2_eval_by_step", []):
            writer.writerow([f"phase2_step{ev['step']}_task1_sr", ev["task1_success_rate"]])
            writer.writerow([f"phase2_step{ev['step']}_task2_sr", ev["task2_success_rate"]])


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
        choices=[
            "self_replay",
            "two_lrs",
            "cotraining",
            "on_policy_self_replay",
            "self_check",
            "no_augment",
        ],
        default="self_replay",
        help="Phase 2 mode: self_replay (task_1 pseudo-labels), two_lrs (same data as self_replay; optimizer 2.5e-4 action expert / 2.5e-5 base, both phases), cotraining, on_policy_self_replay, self_check, no_augment (task_2 only). For two_lrs, --lr is only for run folder naming.",
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
    parser.add_argument(
        "--merging_eps",
        type=float,
        default=1.0,
        help="Merge phase2 with phase1 after phase 2: eps*phase2 + (1-eps)*phase1. Default 1 = keep phase2.",
    )
    parser.add_argument(
        "--alignment_ratio_threshold",
        type=float,
        default=None,
        help="For self_replay, two_lrs, on_policy_self_replay: keep task1 samples with alignment_ratio <= threshold, weight others 0 (default 0.2 when enabled). Omit to disable.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="LR warmup steps: linearly from 0 to lr over this many steps, then constant. Default 0 = no warmup.",
    )
    parser.add_argument(
        "--lambda_kl",
        type=float,
        default=1.0,
        help="For self_check mode: weight for model-predicted pseudo-action half of the batch. Default 1.",
    )
    args = parser.parse_args()

    if args.lora and args.action_expert_only:
        parser.error("--lora and --action_expert_only are mutually exclusive")
    if not (0.0 <= args.merging_eps <= 1.0):
        parser.error("--merging_eps must be in [0, 1]")

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
    merging_eps = args.merging_eps
    alignment_ratio_threshold = args.alignment_ratio_threshold
    warmup_steps = args.warmup_steps
    lambda_kl = args.lambda_kl
    two_lrs = mode == "two_lrs"
    run_dir, video_dir, losses_pdf = _ensure_run_dir(
        task1, task2, seed, mode, lr, num_steps,
        single_episode=single_episode,
        lora=lora,
        action_expert_only=action_expert_only,
        merging_eps=merging_eps,
        alignment_ratio_threshold=alignment_ratio_threshold,
        warmup_steps=warmup_steps,
        lambda_kl=lambda_kl,
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
    if two_lrs:
        print("Optimizer: two LRs — action expert 2.5e-4, rest 2.5e-5 (phase 1 & 2); --lr is for run dir naming only")
    print(f"Merging eps: {merging_eps} (1=keep phase2, 0=keep phase1)")
    print(f"Num episodes: {num_episodes}, Steps: {num_steps}, LR: {lr}, Warmup steps: {warmup_steps}")
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

    lr_action_expert, lr_base = 2.5e-4, 2.5e-5
    results = {
        "mode": mode,
        "merging_eps": merging_eps,
        "warmup_steps": warmup_steps,
        "lambda_kl": lambda_kl if mode == "self_check" else None,
        "lr_action_expert": lr_action_expert if two_lrs else None,
        "lr_base": lr_base if two_lrs else None,
        "phase1_train_losses": [],
        "phase1_train_grad_norms": [],
        "phase1_train_update_norms": [],
        "phase1_train_grad_cosine_sims": [],
        "phase1_eval_task1_success_rate": None,
        "phase2_train_losses": [],
        "phase2_train_grad_norms": [],
        "phase2_train_update_norms": [],
        "phase2_train_grad_cosine_sims": [],
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
    phase1_grad_cosine_sims: list[float] = []

    def _on_step_phase1(step: int, loss_val: float) -> None:
        phase1_losses.append(loss_val)

    def _on_step_info_phase1(step: int, info: dict) -> None:
        phase1_grad_norms.append(info["grad_norm"])
        phase1_update_norms.append(info["update_norm"])
        phase1_grad_cosine_sims.append(info["grad_cosine_sim"])

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
        two_lrs=two_lrs,
        lr_action_expert=lr_action_expert,
        lr_base=lr_base,
    )
    model_phase1.eval()
    results["phase1_train_losses"] = phase1_losses
    results["phase1_train_grad_norms"] = phase1_grad_norms
    results["phase1_train_update_norms"] = phase1_update_norms
    results["phase1_train_grad_cosine_sims"] = phase1_grad_cosine_sims
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
    # Phase 2: Fine-tune on task_2 (mode: self_replay, two_lrs, cotraining, on_policy_self_replay, self_check, or no_augment)
    # -------------------------------------------------------------------------
    phase2_augment = mode in ("self_replay", "two_lrs")
    phase2_cotraining = mode == "cotraining"
    phase2_on_policy_self_replay = mode == "on_policy_self_replay"
    phase2_self_check = mode == "self_check"
    if phase2_cotraining:
        print("\n" + "=" * 50)
        print("Phase 2: Cotraining on task_2 + task_1 (independent sampling)")
        print("=" * 50)
    elif phase2_on_policy_self_replay:
        print("\n" + "=" * 50)
        print("Phase 2: On-policy self-replay (task_2 + task_1 with reference model actions)")
        print("=" * 50)
    elif phase2_self_check:
        print("\n" + "=" * 50)
        print(
            "Phase 2: Self-check on task_2 — pseudo-labels from current model each step, "
            f"lambda_kl={lambda_kl}"
        )
        print("=" * 50)
    elif mode == "two_lrs":
        print("\n" + "=" * 50)
        print(
            "Phase 2: Self-replay on task_2 with task_1 pseudo-labels (two_lrs mode: dual LR in both phases)"
        )
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
        make_pseudo_label_inference_fn(policy_phase1)
        if (phase2_augment or phase2_on_policy_self_replay)
        else None
    )
    if phase2_augment:
        augment_prompt = LIBERO_90_TASK_IDS_PROMPTS.get(task1, "do nothing")
        print(f"Augmentation prompt (task_1={task1}): {augment_prompt!r}")
    if phase2_on_policy_self_replay:
        print(f"On-policy cotraining task_1={task1}: inputs from dataset, actions from phase1 model")
        print(f"Using half batch size ({batch_size // 2}) for on_policy_self_replay")
    if (phase2_on_policy_self_replay or phase2_augment) and alignment_ratio_threshold is not None:
        print(f"Alignment ratio filtering: keep task1 samples with ratio <= {alignment_ratio_threshold}")

    phase2_config = (
        dataclasses.replace(config, batch_size=batch_size // 2)
        if phase2_on_policy_self_replay
        else config
    )
    with override_create_txorch_dataset(
        repo_id=repo_id,
        task_id=task2,
        mirror_data=True,
        single_episode=single_episode,
        augment=phase2_augment,
        augment_task_id=task1 if phase2_augment else None,
        augment_inference_fn=augment_inference_fn,
        cotraining=phase2_cotraining,
        cotraining_task_id=task1 if phase2_cotraining else None,
        on_policy_self_replay=phase2_on_policy_self_replay,
        on_policy_cotraining_task_id=task1 if phase2_on_policy_self_replay else None,
        on_policy_inference_fn=augment_inference_fn if phase2_on_policy_self_replay else None,
    ):
        train_loader = _data_loader.create_data_loader(
            phase2_config, sharding=None, shuffle=True,
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
    phase2_grad_cosine_sims: list[float] = []
    phase2_alignment_avg_ratios: list[float] = []
    phase2_val_task1: list[dict] = []
    phase2_eval_by_step: list[dict] = []

    def _on_step_phase2(step: int, loss_val: float) -> None:
        phase2_train_losses.append(loss_val)

    def _on_step_info_phase2(step: int, info: dict) -> None:
        phase2_grad_norms.append(info["grad_norm"])
        phase2_update_norms.append(info["update_norm"])
        phase2_grad_cosine_sims.append(info["grad_cosine_sim"])
        if "alignment_task1_avg_ratio" in info:
            phase2_alignment_avg_ratios.append(info["alignment_task1_avg_ratio"])

    def _on_val_task1(step: int, loss_val: float) -> None:
        phase2_val_task1.append({"step": step, "loss": loss_val})

    def _on_eval_phase2(step: int, model) -> None:
        """Run env evaluation every evaluation_interval steps during Phase 2."""
        policy = create_policy(model, config, CHECKPOINT_DIR, rng_seed=seed)
        sr_t1, _ = run_evaluation(
            policy=policy,
            train_config=config,
            num_trials=num_episodes,
            task_suite_name=TASK_SUITE_NAME,
            task_id=task1,
            save_video=False,
            video_out_path="",
            seed=seed,
        )
        sr_t2, _ = run_evaluation(
            policy=policy,
            train_config=config,
            num_trials=num_episodes,
            task_suite_name=TASK_SUITE_NAME,
            task_id=task2,
            save_video=False,
            video_out_path="",
            seed=seed,
        )
        phase2_eval_by_step.append({
            "step": step,
            "task1_success_rate": sr_t1,
            "task2_success_rate": sr_t2,
        })
        print(f"  [Phase 2 step {step}] task_1: {sr_t1 * 100:.1f}%, task_2: {sr_t2 * 100:.1f}%")

    model_phase2 = copy_model(model_phase1, config)
    # Use larger validation_interval with grad accumulation to avoid OOM (validation + step boundary)
    phase2_validation_interval = 10 if phase2_on_policy_self_replay else 5
    train_kwargs = dict(
        model=model_phase2,
        training_data_loader=train_loader,
        config=config,
        learning_rate=lr,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        log_interval=max(1, num_steps // 10),
        seed=seed,
        show_progress_bar=True,
        donate_buffers=False,
        on_step_callback=_on_step_phase2,
        on_step_info_callback=_on_step_info_phase2,
        validation_data_loader=val_loader_task1,
        validation_interval=phase2_validation_interval,
        on_validation_callback=_on_val_task1,
        evaluation_interval=20,
        on_evaluation_callback=_on_eval_phase2,
    )
    # Alignment weighting: filter task1 samples by alignment ratio (self_replay and on_policy_self_replay)
    if (
        (phase2_on_policy_self_replay or phase2_augment)
        and alignment_ratio_threshold is not None
    ):
        train_kwargs["alignment_ratio_threshold"] = alignment_ratio_threshold
        train_kwargs["alignment_reference_policy"] = policy_phase1
        train_kwargs["alignment_weight_task1_only"] = True
    # Gradient accumulation for on_policy_self_replay: batch size is halved, so accumulate over 2 steps
    if phase2_on_policy_self_replay:
        train_kwargs["gradient_accumulation_steps"] = 2
    # Self-check: expand each batch inside train_model_on_fly using current model for pseudo-actions
    if phase2_self_check:
        train_kwargs["self_check_lambda_kl"] = lambda_kl
    if two_lrs:
        train_kwargs["two_lrs"] = True
        train_kwargs["lr_action_expert"] = lr_action_expert
        train_kwargs["lr_base"] = lr_base
    model_phase2, _, _ = train_model_on_fly(**train_kwargs)
    model_phase2.eval()

    # Merge phase2 with phase1: eps*phase2 + (1-eps)*phase1 (eps=1 keeps phase2)
    if merging_eps < 1.0:
        print(f"\nMerging models: eps={merging_eps} (blend phase2 with phase1)")
        model_phase2 = merge_model_parameters(
            trained_model=model_phase2,
            original_model=model_phase1,
            merging_eps=merging_eps,
        )

    results["phase2_train_losses"] = phase2_train_losses
    results["phase2_train_grad_norms"] = phase2_grad_norms
    results["phase2_train_update_norms"] = phase2_update_norms
    results["phase2_train_grad_cosine_sims"] = phase2_grad_cosine_sims
    results["phase2_alignment_avg_ratios"] = phase2_alignment_avg_ratios
    results["phase2_val_task1_losses"] = phase2_val_task1
    results["phase2_eval_by_step"] = phase2_eval_by_step

    # -------------------------------------------------------------------------
    # Evaluate on task_1 and task_2 (final eval)
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
        teacher_policy=policy_phase1,
    )
    results["phase2_eval_task1_success_rate"] = sr2_t1
    print(f"Phase 2 eval task_1 success rate: {sr2_t1 * 100:.1f}%")
    teacher_l2_pdf = getattr(run_evaluation, "last_teacher_student_l2_pdf", None)
    if teacher_l2_pdf:
        print(f"Teacher vs student action L2 plot: {teacher_l2_pdf}")

    del policy_phase1, model_phase1

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

    phase2_eval_by_step.append({
        "step": num_steps,
        "task1_success_rate": sr2_t1,
        "task2_success_rate": sr2_t2,
    })

    # -------------------------------------------------------------------------
    # Plot losses (after evals, so final success rates are included)
    # -------------------------------------------------------------------------
    _plot_training_curves_pdf(
        phase1_train_losses=results["phase1_train_losses"],
        phase2_train_losses=results["phase2_train_losses"],
        phase2_val_task1=results["phase2_val_task1_losses"],
        phase1_grad_norms=results["phase1_train_grad_norms"],
        phase2_grad_norms=results["phase2_train_grad_norms"],
        phase1_update_norms=results["phase1_train_update_norms"],
        phase2_update_norms=results["phase2_train_update_norms"],
        phase1_grad_cosine_sims=results["phase1_train_grad_cosine_sims"],
        phase2_grad_cosine_sims=results["phase2_train_grad_cosine_sims"],
        phase2_alignment_avg_ratios=results["phase2_alignment_avg_ratios"],
        phase2_eval_by_step=results["phase2_eval_by_step"],
        pdf_path=losses_pdf,
    )
    print(f"\nSaved losses PDF to {losses_pdf}")

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
