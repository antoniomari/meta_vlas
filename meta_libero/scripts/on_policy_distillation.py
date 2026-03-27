## On-policy distillation: teacher fine-tuned on task1, student trains from base on teacher-labeled rollouts.
#
# 1. Load base Pi0.5 LIBERO model; fine-tune one copy on the task (teacher), then eval teacher
#    for N episodes (default 10) on the same task and record success rate.
# 2. Initialize student from a fresh copy of the base (not teacher weights).
# 3. Loop (max 50 iters): run N student evaluation episodes per iter (default 1); trajectories
#    are merged into one BC dataset; teacher_policy for L2 logging;
#    collect (obs, teacher action chunk, alignment ratio) each replan; optionally filter
#    BC data with --alignment_ratio_threshold (keep ratio <= threshold, thesis / augment convention),
#    or --align_min (keep ratio >= align_min; drops samples with lower alignment scores);
#    BC-train on that episode only unless --cumulative_buffer; stop early when rollout
#    success_rate is 100% (all evaluation episodes in that iter succeed).
#    If --full_experiment: do not stop early on rollout success; every 5 outer iterations run
#    a separate 10-episode student eval (no teacher) and plot that curve alongside rollout SR.
#    If --rollout_episodes 1, after the distillation loop (any exit) run a final student eval
#    on 10 episodes (no teacher logging) and record it in results.csv.
#    Saves videos/iter_XXX/alignment_ratios.pdf per distillation iteration (no JSON).
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

import argparse
import csv
import dataclasses
import gc
import logging
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from openpi.training import data_loader as _data_loader  # type: ignore

from meta_libero.src.dataset import override_create_torch_dataset  # type: ignore
from meta_libero.src.ttt import (  # type: ignore
    NeighborsDataLoader,
    copy_model,
    create_policy,
    load_pi05_libero_model,
    run_evaluation,
    train_model_on_fly,
)

TASK_SUITE_NAME = "libero_90"
# When rollout_episodes==1, run this many trials on the final student after the distillation loop.
STUDENT_FINAL_EVAL_EPISODES = 10
# --full_experiment: eval interval (outer iterations) and episode count for periodic eval.
FULL_EXPERIMENT_EVAL_INTERVAL = 5
FULL_EXPERIMENT_EVAL_EPISODES = 10
CHECKPOINT_DIR = os.getenv(
    "OPENPI_CHECKPOINT_DIR",
    str(Path.home() / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero"),
)


def _results_root() -> Path:
    return Path(os.getenv("META_LIBERO_RESULTS_DIR", "meta_libero/results"))


def _periodic_student_eval_10ep(
    *,
    student_model,
    config,
    task_id: int,
    seed: int,
    outer_iter: int,
    video_dir: Path,
    save_video: bool,
) -> float:
    """10-episode student eval without teacher (for --full_experiment periodic curve)."""
    out = video_dir / "periodic_eval_10ep" / f"iter_{outer_iter:03d}"
    out.mkdir(parents=True, exist_ok=True)
    pol = create_policy(
        student_model,
        config,
        CHECKPOINT_DIR,
        rng_seed=seed + outer_iter * 100_003 + 444_444,
    )
    sr, _ = run_evaluation(
        policy=pol,
        train_config=config,
        num_trials=FULL_EXPERIMENT_EVAL_EPISODES,
        task_suite_name=TASK_SUITE_NAME,
        task_id=task_id,
        save_video=save_video,
        video_out_path=str(out),
        seed=seed + outer_iter + 888_888,
        teacher_policy=None,
        show_progress_bar=True,
    )
    return float(sr)


def _setup_logging() -> None:
    class VersionWarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "is in 2.0 format" not in record.getMessage()

    logging.getLogger().addFilter(VersionWarningFilter())
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logging.getLogger("OpenGL").setLevel(logging.ERROR)


def _ensure_run_dir(
    task_id: int,
    seed: int,
    teacher_lr: float,
    teacher_steps: int,
    bc_lr: float,
    bc_steps: int,
    max_iters: int,
    single_episode: bool = False,
    lora: bool = False,
    action_expert_only: bool = False,
    alignment_ratio_threshold: float | None = None,
    align_min: float | None = None,
    rollout_episodes: int = 1,
    full_experiment: bool = False,
) -> tuple[Path, Path, Path]:
    """Return (run_dir, video_dir, losses_pdf)."""
    folder_base = "on_policy_distillation_single" if single_episode else "on_policy_distillation"
    if lora:
        folder_base = f"{folder_base}_lora"
    elif action_expert_only:
        folder_base = f"{folder_base}_action_expert"
    parent = _results_root() / folder_base / f"task{task_id}_seed{seed}"
    base_name = (
        f"teacher_lr{teacher_lr}_steps{teacher_steps}_bc_lr{bc_lr}_steps{bc_steps}_maxiters{max_iters}"
    )
    if rollout_episodes != 1:
        base_name = f"{base_name}_rolloutEp{rollout_episodes}"
    if alignment_ratio_threshold is not None:
        base_name = f"{base_name}_align{alignment_ratio_threshold}"
    elif align_min is not None:
        base_name = f"{base_name}_alignmin{align_min}"
    if full_experiment:
        base_name = f"{base_name}_full"
    base = parent / base_name
    base.mkdir(parents=True, exist_ok=True)
    video_dir = base / "videos"
    video_dir.mkdir(exist_ok=True)
    losses_pdf = base / "losses.pdf"
    return base, video_dir, losses_pdf


_BC_META_KEYS = frozenset({"alignment_ratio", "replan_env_step"})


def _example_alignment_ratio(ex: dict) -> float:
    v = ex.get("alignment_ratio")
    if v is None:
        return float("nan")
    return float(np.asarray(v, dtype=np.float64).reshape(-1)[0])


def _example_env_step(ex: dict) -> int | None:
    v = ex.get("replan_env_step")
    if v is None:
        return None
    return int(np.asarray(v).reshape(-1)[0])


def _kept_by_alignment_policy(
    ratio: float,
    *,
    alignment_ratio_threshold: float | None,
    align_min: float | None,
) -> bool:
    """Keep sample: either ratio <= threshold (augment convention) or ratio >= align_min."""
    if alignment_ratio_threshold is None and align_min is None:
        return True
    if not math.isfinite(ratio):
        return True
    if alignment_ratio_threshold is not None:
        return ratio <= alignment_ratio_threshold
    assert align_min is not None
    return ratio >= align_min


def _filter_examples_by_alignment(
    examples: list[dict],
    *,
    alignment_ratio_threshold: float | None = None,
    align_min: float | None = None,
) -> tuple[list[dict], int, int]:
    """Return (kept examples, n_kept, n_dropped)."""
    if alignment_ratio_threshold is not None and align_min is not None:
        raise ValueError("only one of alignment_ratio_threshold and align_min may be set")
    if alignment_ratio_threshold is None and align_min is None:
        return list(examples), len(examples), 0
    kept: list[dict] = []
    dropped = 0
    for ex in examples:
        if _kept_by_alignment_policy(
            _example_alignment_ratio(ex),
            alignment_ratio_threshold=alignment_ratio_threshold,
            align_min=align_min,
        ):
            kept.append(ex)
        else:
            dropped += 1
    return kept, len(kept), dropped


def _strip_bc_metadata(ex: dict) -> dict:
    return {k: v for k, v in ex.items() if k not in _BC_META_KEYS}


def _plot_alignment_ratios_iter_pdf(
    *,
    iter_idx: int,
    env_steps: list[int],
    ratios: list[float],
    n_kept: int,
    n_total: int,
    alignment_ratio_threshold: float | None = None,
    align_min: float | None = None,
    pdf_path: Path,
) -> None:
    """One rollout: scatter (env step vs ratio) + histogram; written under e.g. videos/iter_000/."""
    if alignment_ratio_threshold is not None and align_min is not None:
        raise ValueError("only one of alignment_ratio_threshold and align_min for plot")
    ys = [float(y) for y in ratios]
    if not ys:
        return
    xs = env_steps if len(env_steps) == len(ys) else list(range(len(ys)))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    cutoff = (
        alignment_ratio_threshold if alignment_ratio_threshold is not None else align_min
    )
    use_min_mode = align_min is not None
    with PdfPages(pdf_path) as pdf:
        fig1, ax1 = plt.subplots(figsize=(6.5, 4.0))
        colors = [
            "C0"
            if _kept_by_alignment_policy(
                y,
                alignment_ratio_threshold=alignment_ratio_threshold,
                align_min=align_min,
            )
            else "C3"
            for y in ys
        ]
        ax1.scatter(xs, ys, c=colors, s=16, alpha=0.88, edgecolors="none")
        if cutoff is not None:
            ax1.axhline(cutoff, color="0.4", linestyle="--", linewidth=1)
        ax1.set_xlabel("Environment step")
        ax1.set_ylabel("Alignment ratio")
        ax1.set_title(f"Outer iter {iter_idx} — kept {n_kept}/{n_total} for BC")
        ax1.grid(True, alpha=0.3)
        if cutoff is not None:
            if use_min_mode:
                leg = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="C0",
                        markersize=7,
                        label="Kept (ratio >= align_min)",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="C3",
                        markersize=7,
                        label="Dropped (ratio < align_min)",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="0.4",
                        linestyle="--",
                        label=f"align_min {cutoff}",
                    ),
                ]
            else:
                leg = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="C0",
                        markersize=7,
                        label="Kept (ratio <= threshold)",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="C3",
                        markersize=7,
                        label="Dropped (ratio > threshold)",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="0.4",
                        linestyle="--",
                        label=f"Threshold {cutoff}",
                    ),
                ]
            fig1.legend(handles=leg, loc="upper right", fontsize=8, framealpha=0.9)
        plt.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)

        fin = [x for x in ys if math.isfinite(x)]
        if fin:
            fig2, axh = plt.subplots(figsize=(6.5, 3.5))
            axh.hist(
                fin,
                bins=min(30, max(8, len(fin) // 2)),
                color="steelblue",
                edgecolor="white",
                alpha=0.9,
            )
            if cutoff is not None:
                axh.axvline(
                    cutoff,
                    color="darkred",
                    linestyle="--",
                    linewidth=1.5,
                    label=("align_min" if use_min_mode else "threshold") + f"={cutoff}",
                )
                axh.legend(loc="upper right", fontsize=8)
            axh.set_xlabel("Alignment ratio")
            axh.set_ylabel("Count")
            axh.set_title(f"Outer iter {iter_idx} — distribution (this rollout)")
            axh.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)


def _plot_losses_pdf(
    *,
    teacher_losses: list[float],
    bc_losses: list[float],
    bc_step_offsets: list[int],
    bc_loss_runs: list[list[float]],
    bc_run_iters: list[int],
    distill_eval_by_iter: list[dict] | None,
    periodic_eval_10ep_by_iter: list[dict] | None,
    pdf_path: Path,
) -> None:
    """Multi-page PDF: (1) teacher + BC + distill success rate, (2) one panel per BC phase."""
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
        ax0, ax1, ax2 = axes[0], axes[1], axes[2]
        if teacher_losses:
            ax0.plot(
                range(len(teacher_losses)),
                teacher_losses,
                "b-",
                label="Teacher fine-tune",
                linewidth=1.5,
            )
        ax0.set_ylabel("Loss")
        ax0.set_xlabel("Gradient step")
        ax0.set_title("Teacher training loss")
        ax0.legend(loc="upper right", fontsize=8)
        ax0.grid(True, alpha=0.3)

        if bc_losses:
            xs = list(range(len(bc_losses)))
            ax1.plot(xs, bc_losses, "c-", label="Student BC (all phases)", linewidth=1.0)
            for off in bc_step_offsets[1:]:
                if 0 < off < len(bc_losses):
                    ax1.axvline(off - 0.5, color="0.7", linestyle=":", linewidth=0.8)
        ax1.set_xlabel("BC gradient step (cumulative over distillation loop)")
        ax1.set_ylabel("Loss")
        ax1.set_title("Student BC loss (full timeline)")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        if distill_eval_by_iter or periodic_eval_10ep_by_iter:
            if distill_eval_by_iter:
                iters = [int(x["iter"]) for x in distill_eval_by_iter]
                srs = [float(x["success_rate"]) * 100.0 for x in distill_eval_by_iter]
                ax2.plot(
                    iters,
                    srs,
                    "g-o",
                    label="Rollout SR (training eval)",
                    linewidth=1.2,
                    markersize=5,
                )
            if periodic_eval_10ep_by_iter:
                piters = [int(x["iter"]) for x in periodic_eval_10ep_by_iter]
                psrs = [float(x["success_rate"]) * 100.0 for x in periodic_eval_10ep_by_iter]
                ax2.plot(
                    piters,
                    psrs,
                    "m-^",
                    label=f"10-ep eval (every {FULL_EXPERIMENT_EVAL_INTERVAL} iters)",
                    linewidth=1.2,
                    markersize=7,
                )
            ax2.set_ylabel("Success %")
            ax2.set_xlabel("Distillation outer iteration")
            ax2.set_title(
                "Student success rate: rollout vs periodic 10-episode evaluation"
            )
            ax2.set_ylim(-5, 105)
            ax2.legend(loc="upper right", fontsize=8)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        n = len(bc_loss_runs)
        if n == 0:
            return

        ncols = min(5, n)
        nrows = (n + ncols - 1) // ncols
        fig2, axes2 = plt.subplots(
            nrows,
            ncols,
            figsize=(3.0 * ncols, 2.4 * nrows),
            squeeze=False,
        )
        for i, losses in enumerate(bc_loss_runs):
            r, c = divmod(i, ncols)
            ax = axes2[r][c]
            it = bc_run_iters[i] if i < len(bc_run_iters) else i
            if losses:
                ax.plot(range(len(losses)), losses, "c-", linewidth=1.2)
            ax.set_title(f"Student BC (outer iter {it})", fontsize=8)
            ax.set_xlabel("Step", fontsize=7)
            ax.set_ylabel("Loss", fontsize=7)
            ax.grid(True, alpha=0.3)
        for j in range(n, nrows * ncols):
            r, c = divmod(j, ncols)
            axes2[r][c].set_visible(False)
        fig2.suptitle("Student BC loss — each training phase (local steps)", fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)


def _save_results_csv(run_dir: Path, results: dict) -> None:
    """Augment-style metric/value CSV plus one row per distillation-iter success rate."""
    csv_path = run_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["task_id", results.get("task_id")])
        writer.writerow(["seed", results.get("seed")])
        writer.writerow(["rollout_episodes_per_iter", results.get("rollout_episodes")])
        if results.get("rollout_episodes") == 1:
            writer.writerow(
                [
                    "student_final_eval_10ep_success_rate",
                    results.get("student_final_eval_10ep_success_rate"),
                ]
            )
        writer.writerow(["teacher_steps", results.get("teacher_steps")])
        writer.writerow(["teacher_lr", results.get("teacher_lr")])
        writer.writerow(["teacher_eval_num_episodes", results.get("teacher_eval_num_episodes")])
        writer.writerow(["teacher_eval_success_rate", results.get("teacher_eval_success_rate")])
        writer.writerow(["teacher_final_loss", results.get("teacher_final_loss")])
        writer.writerow(["bc_steps_per_iter", results.get("bc_steps_per_iter")])
        writer.writerow(["bc_lr", results.get("bc_lr")])
        writer.writerow(["max_iters", results.get("max_iters")])
        writer.writerow(["cumulative_buffer", results.get("cumulative_buffer")])
        writer.writerow(["alignment_ratio_threshold", results.get("alignment_ratio_threshold")])
        writer.writerow(["align_min", results.get("align_min")])
        writer.writerow(["n_iters_ran", results.get("n_iters_ran")])
        writer.writerow(["final_success", results.get("final_success")])
        writer.writerow(["stopped_reason", results.get("stopped_reason")])
        writer.writerow(["final_bc_buffer_size", results.get("final_bc_buffer_size")])
        writer.writerow(["full_experiment", results.get("full_experiment")])
        for ev in results.get("distill_eval_by_iter", []):
            writer.writerow(
                [f"distill_iter{ev['iter']}_success_rate", ev["success_rate"]]
            )
        for ev in results.get("periodic_eval_10ep_by_iter", []) or []:
            writer.writerow(
                [
                    f"periodic_eval10ep_iter{ev['iter']}_success_rate",
                    ev["success_rate"],
                ]
            )


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(
        description="On-policy distillation: teacher on task1, student BC on teacher-labeled student rollouts"
    )
    parser.add_argument("--task", type=int, required=True, help="LIBERO task id for teacher FT and student eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--teacher_steps", type=int, default=50, help="Fine-tuning steps for teacher on task1")
    parser.add_argument("--teacher_lr", type=float, default=1e-4, help="LR for teacher fine-tuning")
    parser.add_argument("--bc_steps", type=int, default=10, help="BC gradient steps per outer iteration")
    parser.add_argument("--bc_lr", type=float, default=1e-4, help="LR for student BC")
    parser.add_argument(
        "--max_iters",
        type=int,
        default=50,
        help=(
            "Max outer iterations (stop early when rollout success is 100%% unless "
            "--full_experiment)"
        ),
    )
    parser.add_argument(
        "--teacher_eval_episodes",
        type=int,
        default=10,
        help="After teacher fine-tuning, evaluate on the same task for this many episodes",
    )
    parser.add_argument(
        "--rollout_episodes",
        type=int,
        default=1,
        help=(
            "Per outer distillation iteration: run this many student eval rollouts; "
            "all trajectories are merged into one BC dataset (same as num_trials in run_evaluation)."
        ),
    )
    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument("--no_save_video", action="store_false", dest="save_video")
    parser.add_argument(
        "--cumulative_buffer",
        action="store_true",
        help=(
            "Accumulate BC data across iterations. "
            "Default is off: each BC phase uses only the latest rollout."
        ),
    )
    parser.add_argument(
        "--alignment_ratio_threshold",
        type=float,
        default=None,
        help=(
            "Filter BC samples: keep only those with alignment ratio <= threshold "
            "(thesis / augment_finetune convention: low ratio = more prompt-aligned). "
            "Omit to disable filtering. Mutually exclusive with --align_min."
        ),
    )
    parser.add_argument(
        "--align_min",
        type=float,
        default=None,
        help=(
            "Alternative to --alignment_ratio_threshold: keep BC samples with "
            "alignment ratio >= this value (drop samples with lower alignment scores). "
            "Mutually exclusive with --alignment_ratio_threshold."
        ),
    )
    parser.add_argument("--single_episode", action="store_true", help="Single-episode dataset for teacher FT")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--action_expert_only", action="store_true")
    parser.add_argument(
        "--full_experiment",
        action="store_true",
        help=(
            "Do not stop early when rollout success is 100%%; run all max_iters. "
            f"Every {FULL_EXPERIMENT_EVAL_INTERVAL} outer iterations, run a separate "
            f"{FULL_EXPERIMENT_EVAL_EPISODES}-episode student eval (no teacher) and plot it."
        ),
    )
    args = parser.parse_args()
    if args.alignment_ratio_threshold is not None and args.align_min is not None:
        parser.error("use only one of --alignment_ratio_threshold and --align_min")

    task_id = args.task
    seed = args.seed
    batch_size = args.batch_size
    teacher_steps = args.teacher_steps
    teacher_lr = args.teacher_lr
    bc_steps = args.bc_steps
    bc_lr = args.bc_lr
    max_iters = args.max_iters
    repo_id = "antoniomari/libero_90"
    align_th = args.alignment_ratio_threshold
    align_min = args.align_min
    rollout_episodes = max(1, int(args.rollout_episodes))

    run_dir, video_dir, losses_pdf = _ensure_run_dir(
        task_id=task_id,
        seed=seed,
        teacher_lr=teacher_lr,
        teacher_steps=teacher_steps,
        bc_lr=bc_lr,
        bc_steps=bc_steps,
        max_iters=max_iters,
        single_episode=args.single_episode,
        lora=args.lora,
        action_expert_only=args.action_expert_only,
        alignment_ratio_threshold=align_th,
        align_min=align_min,
        rollout_episodes=rollout_episodes,
        full_experiment=args.full_experiment,
    )

    print(f"Run directory: {run_dir}")

    print("\nLoading base model...")
    base_model, config = load_pi05_libero_model(
        use_base_model=False,
        use_lora=args.lora,
        action_expert_only=args.action_expert_only,
    )
    config = dataclasses.replace(config, batch_size=batch_size)

    # --- Teacher: copy(base) -> FT on task ---
    print("\n" + "=" * 50)
    print(f"Teacher: fine-tune on task {task_id}")
    print("=" * 50)
    with override_create_torch_dataset(
        repo_id=repo_id,
        task_id=task_id,
        mirror_data=True,
        single_episode=args.single_episode,
        augment=False,
    ):
        teacher_train_loader = _data_loader.create_data_loader(
            config, sharding=None, shuffle=True,
        )

    teacher_model = copy_model(base_model, config)
    teacher_losses: list[float] = []

    def _on_teacher_step(_step: int, loss_val: float) -> None:
        teacher_losses.append(loss_val)

    teacher_model, _, _ = train_model_on_fly(
        model=teacher_model,
        training_data_loader=teacher_train_loader,
        config=config,
        learning_rate=teacher_lr,
        num_steps=teacher_steps,
        warmup_steps=0,
        weight_decay=0.0,
        log_interval=max(1, teacher_steps // 10),
        seed=seed,
        show_progress_bar=True,
        donate_buffers=False,
        on_step_callback=_on_teacher_step,
    )
    teacher_model.eval()
    del teacher_train_loader

    teacher_policy = create_policy(teacher_model, config, CHECKPOINT_DIR, rng_seed=seed)

    teacher_eval_episodes = max(1, int(args.teacher_eval_episodes))
    teacher_eval_video = video_dir / "teacher_post_train_eval"
    teacher_eval_video.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 50)
    print(
        f"Teacher evaluation: {teacher_eval_episodes} episodes on task {task_id} "
        f"(after fine-tuning)"
    )
    print("=" * 50)
    teacher_eval_sr, _ = run_evaluation(
        policy=teacher_policy,
        train_config=config,
        num_trials=teacher_eval_episodes,
        task_suite_name=TASK_SUITE_NAME,
        task_id=task_id,
        save_video=args.save_video,
        video_out_path=str(teacher_eval_video),
        seed=seed,
        show_progress_bar=True,
    )
    print(
        f"Teacher eval success rate: {teacher_eval_sr * 100:.1f}% "
        f"({int(round(teacher_eval_sr * teacher_eval_episodes))}/{teacher_eval_episodes} episodes)"
    )

    # --- Student: fresh copy from base (not teacher) ---
    print("\n" + "=" * 50)
    print("Student: fresh copy of base model (not teacher)")
    print("=" * 50)
    student_model = copy_model(base_model, config)
    student_model.eval()
    del base_model

    bc_buffer: list[dict] = []
    all_bc_losses: list[float] = []
    bc_offsets: list[int] = [0]
    bc_loss_runs: list[list[float]] = []
    bc_run_iters: list[int] = []
    distill_eval_by_iter: list[dict] = []
    periodic_eval_10ep_by_iter: list[dict] = []
    final_success = False
    stopped_reason = "max_iters"

    if args.full_experiment:
        print(
            f"Full experiment mode: no early stop on rollout success; "
            f"every {FULL_EXPERIMENT_EVAL_INTERVAL} outer iterations run "
            f"{FULL_EXPERIMENT_EVAL_EPISODES}-episode eval (see losses.pdf third panel)."
        )

    if align_th is not None:
        print(
            f"Alignment filtering (max/threshold): keep BC samples with alignment_ratio <= {align_th} "
            "(plots: videos/iter_XXX/alignment_ratios.pdf per rollout)"
        )
    elif align_min is not None:
        print(
            f"Alignment filtering (align_min): keep BC samples with alignment_ratio >= {align_min} "
            "(plots: videos/iter_XXX/alignment_ratios.pdf per rollout)"
        )

    for it in range(max_iters):
        iter_video = video_dir / f"iter_{it:03d}"
        iter_video.mkdir(parents=True, exist_ok=True)
        episode_examples: list[dict] = []

        student_policy = create_policy(student_model, config, CHECKPOINT_DIR, rng_seed=seed + it * 100_003)

        print(
            f"\n--- Distillation iter {it + 1}/{max_iters}: "
            f"rollout ({rollout_episodes} ep{'s' if rollout_episodes != 1 else ''}) ---"
        )
        success_rate, _metrics = run_evaluation(
            policy=student_policy,
            train_config=config,
            num_trials=rollout_episodes,
            task_suite_name=TASK_SUITE_NAME,
            task_id=task_id,
            save_video=args.save_video,
            video_out_path=str(iter_video),
            seed=seed + it,
            teacher_policy=teacher_policy,
            show_progress_bar=True,
            distillation_examples_out=episode_examples,
        )
        # Stop only when every rollout episode succeeded (mean rate == 1.0).
        ep_success = success_rate >= 1.0 - 1e-6
        n_new = len(episode_examples)
        ratios = [_example_alignment_ratio(e) for e in episode_examples]
        env_steps: list[int] = [
            j if (s := _example_env_step(e)) is None else int(s)
            for j, e in enumerate(episode_examples)
        ]
        filtered_ep, n_kept, n_dropped = _filter_examples_by_alignment(
            episode_examples,
            alignment_ratio_threshold=align_th,
            align_min=align_min,
        )
        stripped_for_bc = [_strip_bc_metadata(e) for e in filtered_ep]
        if ratios:
            _plot_alignment_ratios_iter_pdf(
                iter_idx=it,
                env_steps=env_steps,
                ratios=ratios,
                n_kept=n_kept,
                n_total=n_new,
                alignment_ratio_threshold=align_th,
                align_min=align_min,
                pdf_path=iter_video / "alignment_ratios.pdf",
            )
        if args.cumulative_buffer:
            bc_buffer.extend(stripped_for_bc)
        else:
            bc_buffer = list(stripped_for_bc)

        distill_eval_by_iter.append({"iter": it, "success_rate": float(success_rate)})
        buf_note = "cumulative" if args.cumulative_buffer else "last-episode only"
        align_note = (
            f", align kept {n_kept}/{n_new} (drop {n_dropped})"
            if (align_th is not None or align_min is not None)
            else ""
        )
        print(
            f"  Iter {it}: success_rate={success_rate:.2f}, new_samples={n_new}{align_note}, "
            f"BC data ({buf_note})={len(bc_buffer)}"
        )

        if ep_success:
            final_success = True
            if not args.full_experiment:
                stopped_reason = "rollout_success"
                break

        if not bc_buffer:
            if n_new > 0 and (align_th is not None or align_min is not None):
                print("  No BC examples left after alignment filter; skipping train step.")
            else:
                print("  No BC examples collected; skipping train step.")
            if args.full_experiment and (it + 1) % FULL_EXPERIMENT_EVAL_INTERVAL == 0:
                pe_sr = _periodic_student_eval_10ep(
                    student_model=student_model,
                    config=config,
                    task_id=task_id,
                    seed=seed,
                    outer_iter=it,
                    video_dir=video_dir,
                    save_video=args.save_video,
                )
                periodic_eval_10ep_by_iter.append({"iter": it, "success_rate": pe_sr})
                ne = FULL_EXPERIMENT_EVAL_EPISODES
                print(
                    f"  Periodic {ne}-ep eval (full_experiment): {pe_sr * 100:.1f}% "
                    f"({int(round(pe_sr * ne))}/{ne})"
                )
            continue

        data_cfg = config.data.create(config.assets_dirs, config.model)
        bc_bs = min(batch_size, len(bc_buffer))
        bc_loader = NeighborsDataLoader(bc_buffer, bc_bs, data_cfg)

        print(f"  BC train: {bc_steps} steps, lr={bc_lr}, examples={len(bc_buffer)}, batch={bc_bs}")
        bc_loss_chunk: list[float] = []

        def _on_bc_step(_step: int, loss_val: float) -> None:
            bc_loss_chunk.append(loss_val)

        student_model, _, _ = train_model_on_fly(
            model=student_model,
            training_data_loader=bc_loader,
            config=config,
            learning_rate=bc_lr,
            num_steps=bc_steps,
            warmup_steps=0,
            weight_decay=0.0,
            log_interval=max(1, bc_steps // 5),
            seed=seed + it + 777,
            show_progress_bar=True,
            donate_buffers=False,
            on_step_callback=_on_bc_step,
        )
        student_model.eval()
        all_bc_losses.extend(bc_loss_chunk)
        bc_offsets.append(len(all_bc_losses))
        bc_loss_runs.append(list(bc_loss_chunk))
        bc_run_iters.append(it)
        gc.collect()

        if args.full_experiment and (it + 1) % FULL_EXPERIMENT_EVAL_INTERVAL == 0:
            pe_sr = _periodic_student_eval_10ep(
                student_model=student_model,
                config=config,
                task_id=task_id,
                seed=seed,
                outer_iter=it,
                video_dir=video_dir,
                save_video=args.save_video,
            )
            periodic_eval_10ep_by_iter.append({"iter": it, "success_rate": pe_sr})
            ne = FULL_EXPERIMENT_EVAL_EPISODES
            print(
                f"  Periodic {ne}-ep eval (full_experiment): {pe_sr * 100:.1f}% "
                f"({int(round(pe_sr * ne))}/{ne})"
            )

    student_final_eval_10ep_sr: float | None = None
    if rollout_episodes == 1:
        print("\n" + "=" * 50)
        print(
            f"Student final evaluation ({STUDENT_FINAL_EVAL_EPISODES} episodes, "
            "rollout_episodes=1 mode)"
        )
        print("=" * 50)
        final_eval_video = video_dir / f"student_final_eval_{STUDENT_FINAL_EVAL_EPISODES}ep"
        final_eval_video.mkdir(parents=True, exist_ok=True)
        final_eval_policy = create_policy(
            student_model,
            config,
            CHECKPOINT_DIR,
            rng_seed=seed + 91_011_013,
        )
        student_final_eval_10ep_sr, _ = run_evaluation(
            policy=final_eval_policy,
            train_config=config,
            num_trials=STUDENT_FINAL_EVAL_EPISODES,
            task_suite_name=TASK_SUITE_NAME,
            task_id=task_id,
            save_video=args.save_video,
            video_out_path=str(final_eval_video),
            seed=seed + 91_011_013,
            teacher_policy=None,
            show_progress_bar=True,
        )
        ne = STUDENT_FINAL_EVAL_EPISODES
        print(
            f"Student final eval: {student_final_eval_10ep_sr * 100:.1f}% "
            f"({int(round(student_final_eval_10ep_sr * ne))}/{ne} episodes)"
        )

    results = {
        "task_id": task_id,
        "seed": seed,
        "teacher_steps": teacher_steps,
        "teacher_lr": teacher_lr,
        "teacher_eval_num_episodes": teacher_eval_episodes,
        "teacher_eval_success_rate": float(teacher_eval_sr),
        "teacher_final_loss": teacher_losses[-1] if teacher_losses else None,
        "bc_steps_per_iter": bc_steps,
        "bc_lr": bc_lr,
        "max_iters": max_iters,
        "rollout_episodes": rollout_episodes,
        "student_final_eval_10ep_success_rate": student_final_eval_10ep_sr,
        "n_iters_ran": len(distill_eval_by_iter),
        "final_success": final_success,
        "stopped_reason": stopped_reason,
        "cumulative_buffer": args.cumulative_buffer,
        "alignment_ratio_threshold": align_th,
        "align_min": align_min,
        "final_bc_buffer_size": len(bc_buffer),
        "distill_eval_by_iter": distill_eval_by_iter,
        "full_experiment": args.full_experiment,
        "periodic_eval_10ep_by_iter": periodic_eval_10ep_by_iter,
    }
    _save_results_csv(run_dir, results)

    _plot_losses_pdf(
        teacher_losses=teacher_losses,
        bc_losses=all_bc_losses,
        bc_step_offsets=bc_offsets,
        bc_loss_runs=bc_loss_runs,
        bc_run_iters=bc_run_iters,
        distill_eval_by_iter=distill_eval_by_iter or None,
        periodic_eval_10ep_by_iter=(
            periodic_eval_10ep_by_iter if args.full_experiment else None
        ),
        pdf_path=losses_pdf,
    )
    print(
        f"\nSaved losses PDF to {losses_pdf} "
        f"(page 1: teacher + student + distill success rate; page 2+: per-BC-phase)"
    )
    print(f"Alignment ratio PDFs (per rollout): {video_dir}/iter_*/alignment_ratios.pdf")
    print(f"Saved results to {run_dir / 'results.csv'} (includes teacher_eval_success_rate)")


if __name__ == "__main__":
    main()
