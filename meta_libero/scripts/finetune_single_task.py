## Fine-tune and evaluate a pi0.5 model on a single LIBERO task
#
# Structured to mirror meta_libero/scripts/ttt_evaluation.py:
#   - same warning / env-var / logging setup
#   - model loaded via load_pi05_libero_model()
#   - dataset prepared via _prepare_dataset()
#   - evaluation done with run_evaluation_ttt(ttt_num_steps=0)
#
# Supports two ways of specifying hyperparameters:
#   1.  YAML config file:   python finetune_single_task.py --config experiments/ft.yaml
#   2.  CLI flags:          python finetune_single_task.py --lr 1e-4 --batch_size 32
#   3.  Both (CLI overrides YAML): python finetune_single_task.py --config ft.yaml --seed 42

import sys
import logging
import warnings

# Suppress warnings BEFORE any other imports (even os!)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*shape requires ndarray or scalar arguments.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax.core.scope")

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/cluster/home/anmari/.cache/huggingface"
os.environ["HF_LEROBOT_HOME"] = "/cluster/home/anmari/.cache/huggingface/lerobot"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
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

from utils import (  # type: ignore
    create_policy,
    load_pi05_libero_model,
    run_evaluation_ttt,
    train_model_on_fly,
)
from configs import (  # type: ignore
    ExperimentConfig,
    FinetuneConfig,
    EvalConfig,
    ModelConfig,
    add_common_args,
    build_experiment_config,
    save_experiment,
)


# ---------------------------------------------------------------------------
# Helpers (mirrors ttt_evaluation.py)
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    class VersionWarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "is in 2.0 format" not in record.getMessage()

    logging.getLogger().addFilter(VersionWarningFilter())
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logging.getLogger("OpenGL").setLevel(logging.ERROR)


def _prepare_dataset(
    config: _config.TrainConfig,
    dataset_to_use: str = "libero_10",
) -> tuple[Any, _config.TrainConfig]:
    """Create the full LIBERO dataset (same as _prepare_ttt_dataset in ttt_evaluation.py)."""
    if dataset_to_use == "libero_10":
        repo_id = "physical-intelligence/libero"
    else:
        assert dataset_to_use == "libero_90"
        repo_id = "physical-intelligence/libero_90"

    with override_create_torch_dataset(repo_id=repo_id):
        dataloader = _data_loader.create_data_loader(
            config, sharding=None, shuffle=False,
        )
        dataset = dataloader._data_loader._data_loader.dataset  # type: ignore[attr-defined]
    return dataset, config


def _init_nn_fetcher(model: _model.BaseModel, dataset_to_use: str = "libero_10") -> NearestNeighborFetcher:
    """Initialise FAISS nearest-neighbour fetcher (same as ttt_evaluation.py)."""
    if dataset_to_use == "libero_10":
        cache_dir = Path.home() / ".cache" / "libero_unified_faiss"
    else:
        assert dataset_to_use == "libero_90"
        cache_dir = Path.home() / ".cache" / "libero_90_norm"

    modality_str = "_".join(sorted(["image1", "image2", "text"]))
    index_path = cache_dir / f"libero_unified_faiss_index_{modality_str}.index"
    metadata_path = cache_dir / f"libero_unified_faiss_metadata_{modality_str}.pkl"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"FAISS index or metadata not found at {index_path} / {metadata_path}. "
            "Please run build_unified_faiss_index.py first."
        )

    return NearestNeighborFetcher(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        model=model,
    )


def _ensure_finetune_results(
    eval_cfg: EvalConfig,
    ft_cfg: FinetuneConfig,
    model_cfg: ModelConfig,
) -> tuple[Path, Path]:
    """Create results directory and return (csv_path, results_dir)."""
    model_suffix = "_base" if model_cfg.use_base_model else ""
    results_dir = (
        Path("meta_libero") / "results"
        / f"full_finetuning_{eval_cfg.task_suite_name}"
        / f"lr_{ft_cfg.learning_rate}{model_suffix}_b{ft_cfg.batch_size}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{eval_cfg.task_suite_name}_{eval_cfg.task_id}.csv"
    return csv_path, results_dir


def _append_csv_row(
    csv_path: Path,
    step: int,
    success_rate: float,
    is_first: bool = False,
) -> None:
    """Append one evaluation result row to CSV."""
    header = ["train_step", "mean_accuracy"]
    mode = "w" if is_first else "a"
    with csv_path.open(mode, newline="") as f:
        writer = csv.writer(f)
        if is_first:
            writer.writerow(header)
        writer.writerow([step, success_rate])
    print(f"  -> Saved to {csv_path.name}")


# ---------------------------------------------------------------------------
# Evaluation helper (wraps run_evaluation_ttt with ttt_num_steps=0)
# ---------------------------------------------------------------------------

def _evaluate(
    model: _model.BaseModel,
    config: _config.TrainConfig,
    checkpoint_dir: str,
    nn_fetcher: NearestNeighborFetcher,
    dataset: Any,
    eval_cfg: EvalConfig,
) -> float:
    """Create a fresh policy from *model* and evaluate with no TTT (ttt_num_steps=0)."""
    policy = create_policy(model, config, checkpoint_dir, rng_seed=eval_cfg.seed)
    success_rate, _episode_metrics = run_evaluation_ttt(
        policy=policy,
        nn_fetcher=nn_fetcher,
        train_config=config,
        dataset=dataset,
        num_trials=eval_cfg.num_trials,
        task_suite_name=eval_cfg.task_suite_name,
        task_id=eval_cfg.task_id,
        save_video=eval_cfg.save_video,
        video_out_path=eval_cfg.video_out_path,
        seed=eval_cfg.seed,
        ttt_num_steps=0,       # <-- no TTT, pure evaluation
        ttt_frequency=9999,    # effectively disabled
        learning_rate=0.0,
        ttt_k=1,
        reset_policy=True,
    )
    return success_rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune model on a single LIBERO task")

    # Shared flags (--config, model flags, eval flags)
    add_common_args(parser)

    # Finetune-specific CLI flags
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--total_steps", type=int, default=None, help="Total gradient steps")
    parser.add_argument("--eval_interval", type=int, default=None, help="Evaluate every N steps")
    parser.add_argument("--warmup_steps", type=int, default=None, help="LR warmup steps")
    parser.add_argument("--skip-first-eval", action="store_true", default=None, help="Skip evaluation before fine-tuning")

    args = parser.parse_args()

    # ---- Build experiment config from YAML + CLI overrides ----------------
    exp = build_experiment_config(args)
    ft_cfg = exp.finetune
    eval_cfg = exp.eval
    model_cfg = exp.model

    _setup_logging()
    CHECKPOINT_DIR = "/cluster/home/anmari/.cache/openpi/openpi-assets/checkpoints/pi05_libero"

    print("=" * 70)
    print("Fine-tune Single Task")
    print("=" * 70)
    print(f"Task suite : {eval_cfg.task_suite_name}")
    print(f"Task ID    : {eval_cfg.task_id}")
    print(f"Num trials : {eval_cfg.num_trials}")
    print(f"LR         : {ft_cfg.learning_rate}")
    print(f"Batch size : {ft_cfg.batch_size}")
    print(f"Total steps: {ft_cfg.num_steps}")
    print(f"Eval every : {ft_cfg.eval_interval}")
    print(f"Warmup     : {ft_cfg.warmup_steps}")
    print(f"Seed       : {eval_cfg.seed}")
    if args.config:
        print(f"Config     : {args.config}")
    print("=" * 70)

    # ---- Load model -------------------------------------------------------
    model, config = load_pi05_libero_model(
        use_base_model=model_cfg.use_base_model,
        use_lora=model_cfg.use_lora,
        action_expert_only=model_cfg.action_expert_only,
    )

    dataset_to_use = model_cfg.dataset_to_use

    # ---- Prepare dataset (for run_evaluation_ttt) -------------------------
    dataset, config = _prepare_dataset(config, dataset_to_use=dataset_to_use)
    print(f"Dataset size: {len(dataset)} samples")

    # ---- Init NN fetcher (required by run_evaluation_ttt) -----------------
    nn_fetcher = _init_nn_fetcher(model, dataset_to_use=dataset_to_use)
    print("NearestNeighborFetcher initialized")

    # ---- Results paths ----------------------------------------------------
    csv_path, results_dir = _ensure_finetune_results(eval_cfg, ft_cfg, model_cfg)
    video_out_path = results_dir / "videos"
    video_out_path.mkdir(parents=True, exist_ok=True)

    # Point eval_cfg.video_out_path at the actual output directory
    eval_cfg = dataclasses.replace(eval_cfg, video_out_path=str(video_out_path))

    # ---- Save the resolved config for reproducibility ---------------------
    save_experiment(exp, results_dir / "experiment_config.yaml")

    eval_results: list[tuple[int, float]] = []

    # ---- Evaluate before fine-tuning (step 0) -----------------------------
    if not ft_cfg.skip_first_eval:
        print("\n" + "=" * 60)
        print("Evaluating BEFORE fine-tuning (step 0)")
        print("=" * 60)
        success_rate = _evaluate(
            model=model,
            config=config,
            checkpoint_dir=CHECKPOINT_DIR,
            nn_fetcher=nn_fetcher,
            dataset=dataset,
            eval_cfg=eval_cfg,
        )
        eval_results.append((0, success_rate))
        _append_csv_row(csv_path, step=0, success_rate=success_rate, is_first=True)

    # ---- Create training data loader --------------------------------------
    config = dataclasses.replace(config, batch_size=ft_cfg.batch_size)

    if dataset_to_use == "libero_10":
        repo_id = "physical-intelligence/libero"
    else:
        repo_id = "physical-intelligence/libero_90"

    with override_create_torch_dataset(repo_id=repo_id, task_id=eval_cfg.task_id):
        data_loader = _data_loader.create_data_loader(
            config, sharding=None, shuffle=True,
        )

    print(f"\nStarting fine-tuning...")
    print(f"Training hyperparameters:")
    print(f"  Learning rate : {ft_cfg.learning_rate}")
    print(f"  Total steps   : {ft_cfg.num_steps}")
    print(f"  Eval interval : {ft_cfg.eval_interval}")
    print(f"  Batch size    : {ft_cfg.batch_size}")
    print(f"  Warmup steps  : {ft_cfg.warmup_steps}")

    # ---- Training + evaluation loop ---------------------------------------
    trained_model = model
    train_state = None
    train_losses: list[float] = []
    num_chunks = ft_cfg.num_steps // ft_cfg.eval_interval

    for chunk_idx in range(num_chunks):
        current_step = (chunk_idx + 1) * ft_cfg.eval_interval

        print("\n" + "=" * 60)
        print(f"Training steps {current_step - ft_cfg.eval_interval} to {current_step}")
        print("=" * 60)

        train_kwargs: dict[str, Any] = dict(
            model=trained_model,
            training_data_loader=data_loader,
            config=config,
            num_steps=ft_cfg.eval_interval,
            log_interval=ft_cfg.log_interval,
            seed=ft_cfg.seed,
        )
        if train_state is None:
            # First chunk: full init
            train_kwargs.update(
                learning_rate=ft_cfg.learning_rate,
                warmup_steps=ft_cfg.warmup_steps,
                weight_decay=ft_cfg.weight_decay,
            )
        else:
            # Subsequent chunks: resume from previous optimizer state
            train_kwargs.update(
                resume_train_state=train_state,
                resume_losses=train_losses,
            )

        trained_model, train_losses, train_state = train_model_on_fly(**train_kwargs)

        # Evaluate
        print("\n" + "=" * 60)
        print(f"Evaluating after step {current_step}")
        print("=" * 60)
        success_rate = _evaluate(
            model=trained_model,
            config=config,
            checkpoint_dir=CHECKPOINT_DIR,
            nn_fetcher=nn_fetcher,
            dataset=dataset,
            eval_cfg=eval_cfg,
        )
        eval_results.append((current_step, success_rate))
        is_first = len(eval_results) == 1 and not csv_path.exists()
        _append_csv_row(csv_path, step=current_step, success_rate=success_rate, is_first=is_first)

    # ---- Plot losses ------------------------------------------------------
    if train_losses:
        plot_filename = results_dir / f"{eval_cfg.task_suite_name}_{eval_cfg.task_id}_losses.pdf"
        print(f"\nSaving losses plot to {plot_filename}")

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(train_losses)), train_losses, linewidth=0.5, alpha=0.7)
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title(f"Training Loss - {eval_cfg.task_suite_name} Task {eval_cfg.task_id}")
        plt.grid(True, alpha=0.3)

        window_size = min(50, len(train_losses) // 10) if len(train_losses) > 10 else 1
        if window_size > 1:
            smoothed = np.convolve(train_losses, np.ones(window_size) / window_size, mode="valid")
            plt.plot(
                range(window_size - 1, len(train_losses)),
                smoothed, "r-", linewidth=2,
                label=f"Smoothed (window={window_size})",
            )
            plt.legend()

        plt.tight_layout()
        plt.savefig(plot_filename, format="pdf", dpi=150)
        plt.close()
        print(f"Saved losses plot")

    # ---- Plot accuracy vs gradient steps ----------------------------------
    if eval_results:
        acc_plot_filename = results_dir / f"{eval_cfg.task_suite_name}_{eval_cfg.task_id}_accuracy.pdf"
        print(f"\nSaving accuracy plot to {acc_plot_filename}")

        plt.figure(figsize=(10, 6))
        steps, accuracies = zip(*eval_results)
        accuracies_percent = [acc * 100 for acc in accuracies]

        plt.plot(
            steps, accuracies_percent, marker="o", linewidth=2, markersize=8,
            label=f"LR={ft_cfg.learning_rate}", color="#1f77b4",
        )
        plt.fill_between(
            steps,
            [max(0, a - 5) for a in accuracies_percent],
            [min(100, a + 5) for a in accuracies_percent],
            alpha=0.2, color="#1f77b4",
        )

        plt.xlabel("# Gradient Steps", fontsize=12)
        plt.ylabel("Success Rate", fontsize=12)
        plt.title(f"Learning rate = {ft_cfg.learning_rate}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 105)
        plt.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(acc_plot_filename, format="pdf", dpi=150)
        plt.close()
        print(f"Saved accuracy plot")

    # ---- Summary ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Task: {eval_cfg.task_suite_name} task {eval_cfg.task_id}")
    print(f"Total training steps: {ft_cfg.num_steps}")
    print(f"\nEvaluation Results:")
    for step, acc in eval_results:
        print(f"  Step {step:4d}: {acc * 100:.1f}% success rate")
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
