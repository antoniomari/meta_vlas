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
from pathlib import Path

# Make script runnable without manual PYTHONPATH exports.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "meta_libero"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

os.environ["HF_HUB_OFFLINE"] = "1"
DEFAULT_HF_HOME = str(Path.home() / ".cache" / "huggingface")
os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
os.environ.setdefault("HF_LEROBOT_HOME", str(Path(os.environ["HF_HOME"]) / "lerobot"))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

from typing import Any
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
import openpi.shared.download as download  # type: ignore

from meta_libero.src.nn_fetcher import NearestNeighborFetcher  # type: ignore
from meta_libero.src.dataset import (  # type: ignore
    override_create_torch_dataset,
)
from meta_libero.src.ttt import (  # type: ignore
    create_policy,
    load_pi05_libero_model,
    run_evaluation,
    train_model_on_fly,
)
from meta_libero.configs import (  # type: ignore
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

def _results_root() -> Path:
    return Path(os.getenv("META_LIBERO_RESULTS_DIR", "meta_libero/results"))


def _checkpoint_dir() -> str:
    return os.getenv(
        "OPENPI_CHECKPOINT_DIR",
        str(Path.home() / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero"),
    )


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
    task_id: int | None = None,
    mirror_data: bool = False,
) -> tuple[Any, _config.TrainConfig]:
    """Create the full LIBERO dataset (same as _prepare_ttt_dataset in ttt_evaluation.py)."""
    if dataset_to_use == "libero_10":
        repo_id = "physical-intelligence/libero"
    else:
        assert dataset_to_use == "libero_90"
        repo_id = "antoniomari/libero_90"

    with override_create_torch_dataset(
        repo_id=repo_id, task_id=task_id, mirror_data=mirror_data
    ):
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
) -> tuple[Path, Path, Path]:
    """Create results tree and return (summary_csv_path, base_dir, run_dir)."""
    model_suffix = "_base" if model_cfg.use_base_model else ""
    mode_subdir = "lora" if model_cfg.use_lora else ("action_expert_only" if model_cfg.action_expert_only else "full")
    base_dir = (
        _results_root()
        / f"full_finetuning_new_{eval_cfg.task_suite_name}"
        / mode_subdir
        / f"lr_{ft_cfg.learning_rate}{model_suffix}_b{ft_cfg.batch_size}"
    )
    run_dir = base_dir / f"{eval_cfg.task_suite_name}_task_{eval_cfg.task_id}" / f"seed{eval_cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = base_dir / "results_summary.csv"
    return summary_csv_path, base_dir, run_dir


def _append_alignment_rows(
    alignment_csv_path: Path,
    *,
    eval_step: int,
    episode_metrics: list[dict[str, Any]],
) -> None:
    """Append alignment-ratio rows from one evaluation pass."""
    header = ["eval_step", "episode_idx", "episode_step", "alignment_ratio", "success"]
    file_exists = alignment_csv_path.exists()
    with alignment_csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for m in episode_metrics:
            episode_idx = int(m.get("episode_idx", -1))
            success = bool(m.get("success", False))
            for episode_step, ratio in m.get("alignment_ratio_by_step", []):
                writer.writerow([eval_step, episode_idx, int(episode_step), float(ratio), success])


def _append_train_losses_chunk(train_losses_csv_path: Path, *, start_step: int, losses: list[float]) -> None:
    """Append a contiguous chunk of training losses with global train-step index."""
    if not losses:
        return
    header = ["train_step", "loss"]
    file_exists = train_losses_csv_path.exists()
    with train_losses_csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for idx, loss_val in enumerate(losses, start=start_step):
            writer.writerow([idx, float(loss_val)])


def _write_train_losses_pdf(
    train_losses: list[float],
    *,
    pdf_path: Path,
    title: str,
) -> None:
    """Overwrite PDF showing current training losses."""
    if not train_losses:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, linewidth=0.5, alpha=0.7)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    window_size = min(50, len(train_losses) // 10) if len(train_losses) > 10 else 1
    if window_size > 1:
        smoothed = np.convolve(train_losses, np.ones(window_size) / window_size, mode="valid")
        plt.plot(
            range(window_size, len(train_losses) + 1),
            smoothed,
            "r-",
            linewidth=2,
            label=f"Smoothed (window={window_size})",
        )
        plt.legend()

    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf", dpi=150)
    plt.close()


def _prepare_image_for_plot(img: np.ndarray) -> np.ndarray:
    """Convert image tensor to plottable HWC float image in [0, 1]."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    arr = arr.astype(np.float32)
    if arr.min() >= -1.0 and arr.max() <= 1.0:
        arr = (arr + 1.0) / 2.0
    elif arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _save_first_batch_samples_pdf(
    dataset: Any,
    *,
    batch_size: int,
    pdf_path: Path,
    max_samples: int = 6,
) -> None:
    """Save random training batch samples to a PDF.

    Uses direct dataset indexing (single process) to avoid multiprocessing
    worker spawning/pickling issues from iterating the training dataloader.
    """
    if dataset is None:
        return

    n_fetch = max(1, int(batch_size))
    n_fetch = min(n_fetch, len(dataset))
    if n_fetch <= 0:
        return

    rng = np.random.default_rng()
    if n_fetch >= len(dataset):
        indices = np.arange(len(dataset))
    else:
        indices = rng.choice(len(dataset), size=n_fetch, replace=False)
    examples = [dataset[int(i)] for i in indices]
    batch = _data_loader._collate_fn(examples)
    observation = _model.Observation.from_dict(batch)
    if not hasattr(observation, "images"):
        return

    base = observation.images.get("base_0_rgb")
    wrist = observation.images.get("left_wrist_0_rgb")
    if base is None or wrist is None:
        return

    base_np = np.asarray(base)
    wrist_np = np.asarray(wrist)
    if base_np.ndim == 3:
        base_np = base_np[None, ...]
    if wrist_np.ndim == 3:
        wrist_np = wrist_np[None, ...]

    n = min(max_samples, int(base_np.shape[0]), int(wrist_np.shape[0]))
    if n <= 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n), squeeze=False)
    for i in range(n):
        axes[i, 0].imshow(_prepare_image_for_plot(base_np[i]))
        axes[i, 0].set_title(f"Sample {i} - Base")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(_prepare_image_for_plot(wrist_np[i]))
        axes[i, 1].set_title(f"Sample {i} - Wrist")
        axes[i, 1].axis("off")

    plt.tight_layout()
    fig.savefig(pdf_path, format="pdf", dpi=150)
    plt.close(fig)


def _append_summary_csv_row(
    summary_csv_path: Path,
    ft_cfg: FinetuneConfig,
    eval_cfg: EvalConfig,
    model_cfg: ModelConfig,
    train_step: int,
    success_rate: float,
) -> None:
    """Append one evaluation summary row to CSV."""
    header = [
        "lr",
        "train_step",
        "batch_size",
        "seed",
        "task_suite_name",
        "task_id",
        "success_rate",
        "num_trials",
        "use_lora",
        "action_expert_only",
    ]
    file_exists = summary_csv_path.exists()
    with summary_csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            ft_cfg.learning_rate,
            train_step,
            ft_cfg.batch_size,
            eval_cfg.seed,
            eval_cfg.task_suite_name,
            eval_cfg.task_id,
            success_rate,
            eval_cfg.num_trials,
            model_cfg.use_lora,
            model_cfg.action_expert_only,
        ])
    print(f"  -> Appended to {summary_csv_path.name}")


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
) -> tuple[float, list[dict[str, Any]]]:
    """Create a fresh policy from *model* and evaluate with no TTT (ttt_num_steps=0)."""
    policy = create_policy(model, config, checkpoint_dir, rng_seed=eval_cfg.seed)
    success_rate, _episode_metrics = run_evaluation(
        policy=policy,
        train_config=config,
        num_trials=eval_cfg.num_trials,
        task_suite_name=eval_cfg.task_suite_name,
        task_id=eval_cfg.task_id,
        save_video=eval_cfg.save_video,
        video_out_path=eval_cfg.video_out_path,
        seed=eval_cfg.seed,
    )
    return success_rate, _episode_metrics


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
    parser.add_argument("--libero-90-dataset", action="store_true", default=None, help="Use LIBERO 90 dataset")
    parser.add_argument(
        "--no-mirror-data",
        action="store_true",
        default=False,
        help="Disable mirrored dataloader transform.",
    )

    args = parser.parse_args()

    # ---- Build experiment config from YAML + CLI overrides ----------------
    exp = build_experiment_config(args)
    ft_cfg = exp.finetune
    eval_cfg = exp.eval
    model_cfg = exp.model

    _setup_logging()
    CHECKPOINT_DIR = _checkpoint_dir()

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

    if args.libero_90_dataset:
        dataset_to_use = "libero_90"
    else:
        dataset_to_use = "libero_10"
    mirror_data = not args.no_mirror_data

    # ---- Prepare dataset (for run_evaluation_ttt) -------------------------
    dataset, config = _prepare_dataset(
        config,
        dataset_to_use=dataset_to_use,
        task_id=eval_cfg.task_id,
        mirror_data=mirror_data,
    )
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Mirror dataloader data: {mirror_data}")

    # ---- Init NN fetcher (required by run_evaluation_ttt) -----------------
    nn_fetcher = _init_nn_fetcher(model, dataset_to_use=dataset_to_use)
    print("NearestNeighborFetcher initialized")

    # ---- Results paths ----------------------------------------------------
    summary_csv_path, results_dir, run_dir = _ensure_finetune_results(eval_cfg, ft_cfg, model_cfg)
    video_out_path = run_dir / "videos"
    video_out_path.mkdir(parents=True, exist_ok=True)
    alignment_csv_path = run_dir / "alignment_scores.csv"
    train_losses_csv_path = run_dir / "training_losses.csv"
    train_losses_pdf_path = run_dir / "training_losses.pdf"
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Point eval_cfg.video_out_path at the actual output directory
    eval_cfg = dataclasses.replace(eval_cfg, video_out_path=str(video_out_path))

    # ---- Save the resolved config for reproducibility ---------------------
    save_experiment(exp, run_dir / "experiment_config.yaml")

    # ---- Create training data loader --------------------------------------
    config = dataclasses.replace(config, batch_size=ft_cfg.batch_size)

    if dataset_to_use == "libero_10":
        repo_id = "physical-intelligence/libero"
    else:
        repo_id = "antoniomari/libero_90"

    with override_create_torch_dataset(
        repo_id=repo_id, task_id=eval_cfg.task_id, mirror_data=mirror_data
    ):
        data_loader = _data_loader.create_data_loader(
            config, sharding=None, shuffle=True,
        )

    eval_results: list[tuple[int, float]] = []

    # ---- Evaluate before fine-tuning (step 0) -----------------------------
    skip_first_eval = True
    if not skip_first_eval:
        print("\n" + "=" * 60)
        print("Evaluating BEFORE fine-tuning (step 0)")
        print("=" * 60)
        success_rate, episode_metrics = _evaluate(
            model=model,
            config=config,
            checkpoint_dir=CHECKPOINT_DIR,
            nn_fetcher=nn_fetcher,
            dataset=dataset,
            eval_cfg=eval_cfg,
        )
        eval_results.append((0, success_rate))
        _append_alignment_rows(
            alignment_csv_path,
            eval_step=0,
            episode_metrics=episode_metrics,
        )
        _append_summary_csv_row(
            summary_csv_path=summary_csv_path,
            ft_cfg=ft_cfg,
            eval_cfg=eval_cfg,
            model_cfg=model_cfg,
            train_step=0,
            success_rate=success_rate,
        )

    # Save first batch of samples
    _save_first_batch_samples_pdf(
        dataset,
        batch_size=ft_cfg.batch_size,
        pdf_path=samples_dir / "samples_0.pdf",
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

        prev_loss_count = len(train_losses)
        trained_model, train_losses, train_state = train_model_on_fly(**train_kwargs)
        new_losses = train_losses[prev_loss_count:]
        _append_train_losses_chunk(
            train_losses_csv_path,
            start_step=prev_loss_count + 1,
            losses=new_losses,
        )
        _write_train_losses_pdf(
            train_losses,
            pdf_path=train_losses_pdf_path,
            title=f"Training Loss - {eval_cfg.task_suite_name} Task {eval_cfg.task_id}",
        )

        # Evaluate
        print("\n" + "=" * 60)
        print(f"Evaluating after step {current_step}")
        print("=" * 60)
        success_rate, episode_metrics = _evaluate(
            model=trained_model,
            config=config,
            checkpoint_dir=CHECKPOINT_DIR,
            nn_fetcher=nn_fetcher,
            dataset=dataset,
            eval_cfg=eval_cfg,
        )
        eval_results.append((current_step, success_rate))
        _append_alignment_rows(
            alignment_csv_path,
            eval_step=current_step,
            episode_metrics=episode_metrics,
        )
        _append_summary_csv_row(
            summary_csv_path=summary_csv_path,
            ft_cfg=ft_cfg,
            eval_cfg=eval_cfg,
            model_cfg=model_cfg,
            train_step=current_step,
            success_rate=success_rate,
        )
        _save_first_batch_samples_pdf(
            dataset,
            batch_size=ft_cfg.batch_size,
            pdf_path=samples_dir / f"samples_{current_step}.pdf",
        )

    # ---- Plot losses ------------------------------------------------------
    if train_losses:
        print(f"\nSaved incremental losses to {train_losses_csv_path}")
        print(f"Saved/updated losses PDF at {train_losses_pdf_path}")

    # ---- Plot accuracy vs gradient steps ----------------------------------
    if eval_results:
        acc_plot_filename = run_dir / "accuracy.pdf"
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
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
