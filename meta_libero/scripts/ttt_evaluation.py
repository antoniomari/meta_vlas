## TTT Evaluation Script
#
# Supports two ways of specifying hyperparameters:
#   1.  YAML config file:   python ttt_evaluation.py --config experiments/my_run.yaml
#   2.  CLI flags:          python ttt_evaluation.py --lr 1e-4 --ttt_k 20
#   3.  Both (CLI overrides YAML): python ttt_evaluation.py --config exp.yaml --seed 42

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

# os.environ["HF_HUB_OFFLINE"] = "1"
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
import jax.numpy as jnp
import jax

from openpi.training import data_loader as _data_loader  # type: ignore
import openpi.models.model as _model  # type: ignore
import openpi.training.config as _config  # type: ignore
import openpi.shared.download as download  # type: ignore
from huggingface_hub import HfApi  # type: ignore

from meta_libero.src.nn_fetcher import NearestNeighborFetcher  # type: ignore
from meta_libero.src.dataset import override_create_torch_dataset  # type: ignore
from meta_libero.src.ttt import (  # type: ignore
    create_policy,
    run_evaluation_ttt,
    run_evaluation,
    run_evaluation_noise,
    load_pi05_libero_model,
)
from meta_libero.configs import (  # type: ignore
    ExperimentConfig,
    TTTConfig,
    EvalConfig,
    ModelConfig,
    NoiseConfig,
    add_common_args,
    build_experiment_config,
    save_experiment,
)


# ---------------------------------------------------------------------------
# Helpers
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
        def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
            return "is in 2.0 format" not in record.getMessage()

    logging.getLogger().addFilter(VersionWarningFilter())
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logging.getLogger("OpenGL").setLevel(logging.ERROR)


def _prepare_ttt_dataset(
    config: _config.TrainConfig,
    dataset_to_use: str = "libero_10"
) -> tuple[Any, _config.TrainConfig]:

    if dataset_to_use == "libero_10":
        repo_id = "antoniomari/libero"
    else:
        assert dataset_to_use == "libero_90"
        repo_id = "antoniomari/libero_90"

    with override_create_torch_dataset(repo_id=repo_id):
        dataloader = _data_loader.create_data_loader(
            config,
            sharding=None,
            shuffle=False,
        )
        dataset = dataloader._data_loader._data_loader.dataset  # type: ignore[attr-defined]
    return dataset, config


def _hf_preflight_check(dataset_to_use: str) -> None:
    """Validate Hugging Face auth and dataset accessibility in job environment."""
    if dataset_to_use == "libero_10":
        repo_id = "antoniomari/libero"
    else:
        assert dataset_to_use == "libero_90"
        repo_id = "antoniomari/libero_90"

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)

    print("=== Hugging Face preflight ===")
    print(f"HF_HOME: {os.getenv('HF_HOME', '<unset>')}")
    print(f"HF token env present: {'yes' if token else 'no (using cached auth if available)'}")

    try:
        who = api.whoami(token=token) if token else api.whoami()
        print(f"HF identity: {who.get('name', '<unknown>')}")
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face authentication failed in this runtime. "
            "Run `huggingface-cli login` or pass HF_TOKEN/HUGGINGFACE_HUB_TOKEN to the job."
        ) from exc

    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
        print(f"HF dataset access OK: {repo_id}")
    except Exception as exc:
        raise RuntimeError(
            f"Hugging Face dataset access failed for `{repo_id}`. "
            "Check repo id, permissions/gating, and account access."
        ) from exc
    print("=== Hugging Face preflight passed ===")


def _init_nn_fetcher(model: _model.BaseModel, dataset_to_use: str = "libero_10") -> NearestNeighborFetcher:

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

    nn_fetcher = NearestNeighborFetcher(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        model=model,
    )
    return nn_fetcher


def _hyperparams_subfolder(ttt_cfg: TTTConfig, eval_cfg: EvalConfig) -> str:
    """Build a filesystem-safe subfolder name from TTT hyperparameters."""
    lr_str = f"{ttt_cfg.learning_rate:.2e}".replace("-0", "-").replace("+0", "")

    if ttt_cfg.noise_ttt:
        return f"noise{lr_str}_freq{ttt_cfg.ttt_frequency}_steps{ttt_cfg.ttt_num_steps}_seed{eval_cfg.seed}"
    else:
        return f"lr{lr_str}_freq{ttt_cfg.ttt_frequency}_steps{ttt_cfg.ttt_num_steps}_k{ttt_cfg.k}_seed{eval_cfg.seed}"


def _ensure_results_paths(
    eval_cfg: EvalConfig,
    ttt_cfg: TTTConfig,
    model_cfg: ModelConfig,
    dataset_to_use: str,
) -> tuple[Path, Path, Path, Path, Path]:
    """Create results directory tree and return (summary_csv, run_dir, losses_pdf, actions_pdf, video_dir)."""
    main_dir = "ttt_base_model" if model_cfg.use_base_model else "ttt"

    if dataset_to_use == "libero_90":
        main_dir += "/dataset_libero_90"
    else:
        main_dir += "/dataset_libero_10"

    if not ttt_cfg.reset_policy:
        main_dir += "_no_reset_policy"

    if model_cfg.use_lora:
        base_dir = _results_root() / main_dir / "lora"
    elif model_cfg.action_expert_only:
        base_dir = _results_root() / main_dir / "action_only"
    else:
        base_dir = _results_root() / main_dir

    task_dir = base_dir / f"{eval_cfg.task_suite_name}_task_{eval_cfg.task_id}"
    hyper_sub = _hyperparams_subfolder(ttt_cfg, eval_cfg)
    run_dir = task_dir / hyper_sub
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = base_dir / "results_summary.csv"
    losses_pdf = run_dir / "losses_grid.pdf"
    actions_pdf = run_dir / "action_distances_grid.pdf"
    video_out_path = run_dir / "videos"
    return summary_csv_path, run_dir, losses_pdf, actions_pdf, video_out_path


def _append_summary_csv_row(
    summary_csv_path: Path,
    ttt_cfg: TTTConfig,
    eval_cfg: EvalConfig,
    model_cfg: ModelConfig,
    batch_size: int | None,
    success_rate: float,
) -> None:
    header = [
        "lr",
        "ttt_frequency",
        "ttt_num_steps",
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
            ttt_cfg.learning_rate,
            ttt_cfg.ttt_frequency,
            ttt_cfg.ttt_num_steps,
            batch_size,
            eval_cfg.seed,
            eval_cfg.task_suite_name,
            eval_cfg.task_id,
            success_rate,
            eval_cfg.num_trials,
            model_cfg.use_lora,
            model_cfg.action_expert_only,
        ])


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
        losses_flat = [val for sub in losses_per_ttt for val in sub]

        if losses_flat:
            ax.plot(range(len(losses_flat)), losses_flat, linewidth=0.8)

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

        success = m.get("success", False)
        status = "Success" if success else "Failure"
        ep_num = m.get("episode_idx", idx) + 1
        ax.set_title(f"Ep {ep_num} ({status})", fontsize=8)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(pdf_path, format="pdf", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run TTT evaluation on a single LIBERO task")

    # Shared flags (--config, model flags, eval flags)
    add_common_args(parser)

    # TTT-specific CLI flags
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for TTT")
    parser.add_argument("--ttt_frequency", type=int, default=None, help="Perform TTT every N steps during rollout")
    parser.add_argument("--ttt_num_steps", type=int, default=None, help="Number of gradient steps per TTT update")
    parser.add_argument("--ttt_k", type=int, default=None, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--max_ttt_step", type=int, default=None, help="Maximum step to perform TTT")
    parser.add_argument("--no-reset-policy", action="store_true", default=None, help="Do not reset policy between episodes")
    parser.add_argument("--noise-ttt", action="store_true", default=None, help="Perform TTT with noise perturbation")
    parser.add_argument("--libero-90-dataset", action="store_true", default=None, help="Use LIBERO 90 dataset")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of noise samples used when generating action chunks for TTT logging/plots. "
        "If > 1, batched inference is used automatically.",
    )
    parser.add_argument(
        "--debug-metrics",
        action="store_true",
        default=False,
        help="Enable expensive per-step auxiliary denoising-loss logging.",
    )

    args = parser.parse_args()

    # ---- Build experiment config from YAML + CLI overrides ----------------
    exp = build_experiment_config(args)
    ttt_cfg = exp.ttt
    eval_cfg = exp.eval
    model_cfg = exp.model

    _setup_logging()
    CHECKPOINT_DIR = _checkpoint_dir()

    print("=" * 70)
    print("TTT Evaluation")
    print("=" * 70)
    print(f"Task suite : {eval_cfg.task_suite_name}")
    print(f"Task ID    : {eval_cfg.task_id}")
    print(f"Num trials : {eval_cfg.num_trials}")
    print(f"LR: {ttt_cfg.learning_rate}, TTT frequency: {ttt_cfg.ttt_frequency}, "
          f"TTT steps: {ttt_cfg.ttt_num_steps}, k: {ttt_cfg.k}")
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

    dataset_to_use = "libero_90" if args.libero_90_dataset else "libero_10"

    _hf_preflight_check(dataset_to_use)

    # ---- Prepare dataset + NN fetcher -------------------------------------
    dataset, config = _prepare_ttt_dataset(config, dataset_to_use=dataset_to_use)
    print(f"TTT dataset size: {len(dataset)} samples")

    nn_fetcher = _init_nn_fetcher(model, dataset_to_use=dataset_to_use)
    print("NearestNeighborFetcher initialized")

    # ---- Create policy ----------------------------------------------------
    policy_ttt = create_policy(
        model, config, CHECKPOINT_DIR, rng_seed=eval_cfg.seed,
    )

    # ---- Results paths ----------------------------------------------------
    summary_csv_path, run_dir, losses_pdf, actions_pdf, video_out_path = _ensure_results_paths(
        eval_cfg, ttt_cfg, model_cfg, dataset_to_use,
    )

    # ---- Save the resolved config for reproducibility ---------------------
    save_experiment(exp, run_dir / "experiment_config.yaml")

    # ---- Run evaluation ---------------------------------------------------
    if ttt_cfg.noise_ttt:
        success_rate, all_episode_metrics = run_evaluation_noise(
            policy=policy_ttt,
            nn_fetcher=nn_fetcher,
            train_config=config,
            dataset=dataset,
            num_trials=eval_cfg.num_trials,
            task_suite_name=eval_cfg.task_suite_name,
            task_id=eval_cfg.task_id,
            save_video=eval_cfg.save_video,
            video_out_path=str(video_out_path),
            seed=eval_cfg.seed,
            noise_frequency=ttt_cfg.ttt_frequency,
            noise_sigma=ttt_cfg.learning_rate,
            max_noise_step=ttt_cfg.ttt_max_step,
        )
    else:
        success_rate, all_episode_metrics = run_evaluation_ttt(
            policy=policy_ttt,
            nn_fetcher=nn_fetcher,
            train_config=config,
            dataset=dataset,
            num_trials=eval_cfg.num_trials,
            task_suite_name=eval_cfg.task_suite_name,
            task_id=eval_cfg.task_id,
            save_video=eval_cfg.save_video,
            video_out_path=str(video_out_path),
            seed=eval_cfg.seed,
            ttt_num_steps=ttt_cfg.ttt_num_steps,
            ttt_frequency=ttt_cfg.ttt_frequency,
            learning_rate=ttt_cfg.learning_rate,
            ttt_k=ttt_cfg.k,
            ttt_use_modalities=ttt_cfg.use_modalities or ["image1", "image2", "text"],
            reset_policy=ttt_cfg.reset_policy,
            max_ttt_step=ttt_cfg.ttt_max_step,
            num_samples=args.num_samples,
            debug_metrics=args.debug_metrics,
        )

    print(f"\nSuccess rate: {success_rate * 100:.1f}%")
    print(f"All episode metrics: {all_episode_metrics}")

    # ---- Persist results --------------------------------------------------
    _append_summary_csv_row(
        summary_csv_path=summary_csv_path,
        ttt_cfg=ttt_cfg,
        eval_cfg=eval_cfg,
        model_cfg=model_cfg,
        batch_size=getattr(config, "batch_size", None),
        success_rate=success_rate,
    )

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
