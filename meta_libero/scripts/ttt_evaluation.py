## TTT Evaluation Script

# NOTE: This script is intentionally structured similarly to
# meta_libero/scripts/finetune_single_task.py but runs only test-time
# training (TTT) evaluation via utils.run_evaluation_ttt.

import sys
import logging
import warnings

# Suppress warnings BEFORE any other imports (even os!)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*shape requires ndarray or scalar arguments.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax.core.scope")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

from pathlib import Path
from typing import Any
import argparse
import csv

import dataclasses

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import jax.numpy as jnp
import jax

sys.path.append("./meta_libero")

from nn_fetcher import NearestNeighborFetcher  # type: ignore
from libero_dataset import override_create_torch_dataset  # type: ignore
from openpi.training import data_loader as _data_loader  # type: ignore
import openpi.models.model as _model  # type: ignore
import openpi.training.config as _config  # type: ignore
import openpi.shared.download as download  # type: ignore

from utils import create_policy, run_evaluation_ttt, load_pi05_libero_model  # type: ignore


def _setup_logging() -> None:
    class VersionWarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
            return "is in 2.0 format" not in record.getMessage()

    logging.getLogger().addFilter(VersionWarningFilter())
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logging.getLogger("OpenGL").setLevel(logging.ERROR)


def _prepare_ttt_dataset(config: _config.TrainConfig, repo_id: str = "physical-intelligence/libero", task_index: int | None = None) -> tuple[Any, _config.TrainConfig]:
    """Create a small dataloader and extract the underlying dataset for TTT."""
    from libero_dataset import override_create_torch_dataset  # local import to avoid cycles

    cfg = dataclasses.replace(config, batch_size=1)
    data_loader = _data_loader.create_data_loader(
        cfg,
        sharding=None,
        shuffle=False,
    )
    dataset = data_loader._data_loader._data_loader.dataset  # type: ignore[attr-defined]
    return dataset, cfg


def _init_nn_fetcher(model: _model.BaseModel) -> NearestNeighborFetcher:
    """Initialize NearestNeighborFetcher using the unified FAISS index."""
    cache_dir = Path.home() / ".cache" / "libero_unified_faiss"
    modality_str = "_".join(sorted(["image1", "image2", "text"]))
    index_path = cache_dir / f"libero_unified_faiss_index_{modality_str}.index"
    metadata_path = cache_dir / f"libero_unified_faiss_metadata_{modality_str}.pkl"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"FAISS index or metadata not found at {index_path} / {metadata_path}. "
            "Please run build_unified_faiss_index.py first."
        )

    nn_fetcher = NearestNeighborFetcher(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        model=model,
    )
    return nn_fetcher


def _ensure_results_paths(task_suite_name: str, task_id: int, action_expert_only: bool = False, use_lora: bool = False) -> tuple[Path, Path, Path]:
    """Create results/ttt/<suite>_task_<id> directory and return CSV + PDF paths.

    Args:
        task_suite_name: Name of the task suite
        task_id: Task ID within the suite
        action_expert_only: If True, use 'ttt/action_only' instead of 'ttt'
        use_lora: If True, use 'ttt/lora' instead of 'ttt' (takes precedence over action_expert_only)
    """
    # Determine base directory based on flags
    if use_lora:
        base_dir = Path("meta_libero") / "results" / "ttt" / "lora"
    elif action_expert_only:
        base_dir = Path("meta_libero") / "results" / "ttt" / "action_only"
    else:
        base_dir = Path("meta_libero") / "results" / "ttt"

    task_dir = base_dir / f"{task_suite_name}_task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)

    csv_path = task_dir / "results.csv"
    losses_pdf = task_dir / "losses_grid.pdf"
    actions_pdf = task_dir / "action_distances_grid.pdf"
    return csv_path, losses_pdf, actions_pdf


def _append_csv_row(
    csv_path: Path,
    lr: float,
    ttt_frequency: int,
    seed: int,
    success_rate: float,
    num_trials: int,
    ttt_num_steps: int,
    ttt_k: int,
) -> None:
    header = ["lr", "ttt_frequency", "seed", "num_trials", "ttt_num_steps", "ttt_k", "success_rate"]
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(
            [
                lr,
                ttt_frequency,
                seed,
                num_trials,
                ttt_num_steps,
                ttt_k,
                success_rate,
            ]
        )


def _plot_grid_losses(episode_metrics: list[dict[str, Any]], pdf_path: Path, max_rows: int = 5, max_cols: int = 10) -> None:
    """Create / overwrite a grid PDF with per-episode loss curves."""
    if not episode_metrics:
        return

    n_cells = min(len(episode_metrics), max_rows * max_cols)
    fig, axes = plt.subplots(max_rows, max_cols, figsize=(20, 10), squeeze=False)

    for idx in range(max_rows * max_cols):
        ax = axes[idx // max_cols][idx % max_cols]
        if idx >= n_cells:
            ax.axis("off")
            continue

        m = episode_metrics[idx]
        losses_per_ttt: list[list[float]] = m.get("losses", [])
        # Flatten all TTT updates within this episode.
        losses_flat = [val for sub in losses_per_ttt for val in sub]

        if losses_flat:
            ax.plot(range(len(losses_flat)), losses_flat, linewidth=0.8)

        # Add success/failure status to title
        success = m.get("success", False)
        status = "Success" if success else "Failure"
        ep_num = m.get("episode_idx", idx) + 1
        ax.set_title(f"Ep {ep_num} ({status})", fontsize=8)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(pdf_path, format="pdf", dpi=150)
    plt.close(fig)


def _plot_grid_action_distances(episode_metrics: list[dict[str, Any]], pdf_path: Path, max_rows: int = 5, max_cols: int = 10) -> None:
    """Create / overwrite a grid PDF with per-episode action-distance curves."""
    if not episode_metrics:
        return

    n_cells = min(len(episode_metrics), max_rows * max_cols)
    fig, axes = plt.subplots(max_rows, max_cols, figsize=(20, 10), squeeze=False)

    for idx in range(max_rows * max_cols):
        ax = axes[idx // max_cols][idx % max_cols]
        if idx >= n_cells:
            ax.axis("off")
            continue

        m = episode_metrics[idx]
        distances_actions = m.get("distances_actions", [])
        if distances_actions:
            xs = [x[0] for x in distances_actions]
            ys = [x[1] for x in distances_actions]
            ax.plot(xs, ys, linewidth=0.8)

        # Add success/failure status to title
        success = m.get("success", False)
        status = "Success" if success else "Failure"
        ep_num = m.get("episode_idx", idx) + 1
        ax.set_title(f"Ep {ep_num} ({status})", fontsize=8)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(pdf_path, format="pdf", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TTT evaluation on a single LIBERO task")
    parser.add_argument("--task_suite_name", type=str, default="libero_10", help="Task suite name (e.g., libero_10, libero_90)")
    parser.add_argument("--task_id", type=int, default=8, help="Task ID within the suite")
    parser.add_argument("--num_trials", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--lr", type=float, default=2.5e-5, help="Learning rate for TTT")
    parser.add_argument("--ttt_frequency", type=int, default=50, help="Perform TTT every N steps during rollout")
    parser.add_argument("--ttt_num_steps", type=int, default=1, help="Number of gradient steps per TTT update")
    parser.add_argument("--ttt_k", type=int, default=1, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--use-base-model", action="store_true", help="Use base pi0.5 weights instead of Libero-pretrained")
    parser.add_argument("--save-video", action="store_true", help="Save rollout videos")
    parser.add_argument("--action-expert-only", action="store_true", help="Only finetune action expert")
    parser.add_argument("--use-lora", action="store_true", help="Use LORA weights instead of base weights")

    args = parser.parse_args()

    _setup_logging()
    CHECKPOINT_DIR = "/cluster/home/anmari/.cache/openpi/openpi-assets/checkpoints/pi05_libero"

    print("=" * 70)
    print("TTT Evaluation")
    print("=" * 70)
    print(f"Task suite: {args.task_suite_name}")
    print(f"Task ID: {args.task_id}")
    print(f"Num trials: {args.num_trials}")
    print(f"LR: {args.lr}, TTT frequency: {args.ttt_frequency}, TTT steps: {args.ttt_num_steps}, k: {args.ttt_k}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    model, config = load_pi05_libero_model(use_lora=args.use_lora, action_expert_only=args.action_expert_only)

    # Prepare dataset for TTT (same repo and task index convention as other scripts).
    dataset, config = _prepare_ttt_dataset(config, repo_id="physical-intelligence/libero", task_index=args.task_id)
    print(f"TTT dataset size: {len(dataset)} samples")

    # Initialize FAISS nearest-neighbor fetcher.
    nn_fetcher = _init_nn_fetcher(model)
    print("✓ NearestNeighborFetcher initialized")

    # Create policy compatible with libero env.
    policy_ttt = create_policy(
        model,
        config,
        CHECKPOINT_DIR,
        rng_seed=args.seed,
    )

    # Run TTT evaluation
    success_rate = run_evaluation_ttt(
        policy=policy_ttt,
        nn_fetcher=nn_fetcher,
        train_config=config,
        dataset=dataset,
        num_trials=args.num_trials,
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        save_video=args.save_video,
        seed=args.seed,
        ttt_num_steps=args.ttt_num_steps,
        ttt_frequency=args.ttt_frequency,
        learning_rate=args.lr,
        ttt_k=args.ttt_k,
        ttt_use_modalities=["image1", "image2", "text"],
    )

    print(f"\nSuccess rate: {success_rate * 100:.1f}%")

    # Prepare per-task outputs
    csv_path, losses_pdf, actions_pdf = _ensure_results_paths(
        args.task_suite_name,
        args.task_id,
        action_expert_only=args.action_expert_only,
        use_lora=args.use_lora
    )
    _append_csv_row(
        csv_path,
        lr=args.lr,
        ttt_frequency=args.ttt_frequency,
        seed=args.seed,
        success_rate=success_rate,
        num_trials=args.num_trials,
        ttt_num_steps=args.ttt_num_steps,
        ttt_k=args.ttt_k,
    )

    # Extract per-episode metrics that run_evaluation_ttt stored on itself.
    episode_metrics = getattr(run_evaluation_ttt, "last_episode_metrics", None)
    if isinstance(episode_metrics, list):
        _plot_grid_losses(episode_metrics, losses_pdf)
        _plot_grid_action_distances(episode_metrics, actions_pdf)
        print(f"Saved losses grid to {losses_pdf}")
        print(f"Saved action-distance grid to {actions_pdf}")
    else:
        print("Warning: run_evaluation_ttt did not expose per-episode metrics; skipping PDF generation.")


if __name__ == "__main__":  # pragma: no cover
    main()

