## On-policy distillation: teacher fine-tuned on task1, student trains from base on teacher-labeled rollouts.
#
# 1. Load base Pi0.5 LIBERO model; fine-tune one copy on the task (teacher), then eval teacher
#    for N episodes (default 10) on the same task and record success rate.
# 2. Initialize student from a fresh copy of the base (not teacher weights). Optional: offline SFT
#    on a second single-episode demo of the same task (--student_pretraining_steps, default 100)
#    with periodic eval before the distillation loop.
# 3. Loop (max 50 iters):
#    run N student evaluation episodes per iter (default 1); trajectories
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
#    Saves videos/iter_XXX/distillation_rollout_metrics.pdf per distillation iteration:
#    teacher alignment ratios, teacher–student action L2, student chunk variance, teacher chunk variance.
#    Use --group_size G>1 for multi-sample student variance; --teacher_group_size T>1 for teacher variance.
#    --grpo_like: per-step group of G student chunks, advantages from -L2 to teacher, weighted diffusion BC.
#    Optional --max_teacher_variance: drop BC rows whose teacher chunk variance (mean_{h,d} var across
#    teacher samples) exceeds the cap; metadata is stored per-row from the rollout (ttt.observation_actions_to_bc_example).
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
from typing import Any

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
    student_action_merge: float = 0.0,
    group_size: int = 1,
    teacher_group_size: int = 1,
    temporal_decay: float = 1.0,
    l1_bc_loss: bool = False,
    kl_lambda: float = 0.0,
    max_teacher_variance: float | None = None,
    student_pretraining_steps: int = 0,
    grpo_like: bool = False,
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
    if abs(float(student_action_merge)) > 1e-12:
        base_name = f"{base_name}_sam{student_action_merge:g}"
    if int(group_size) != 1:
        base_name = f"{base_name}_g{int(group_size)}"
    if int(teacher_group_size) != 1:
        base_name = f"{base_name}_tg{int(teacher_group_size)}"
    if abs(float(temporal_decay) - 1.0) > 1e-12:
        base_name = f"{base_name}_td{temporal_decay:g}"
    if l1_bc_loss:
        base_name = f"{base_name}_l1bc"
    if abs(float(kl_lambda)) > 1e-12:
        base_name = f"{base_name}_kl{kl_lambda:g}"
    if max_teacher_variance is not None:
        base_name = f"{base_name}_mtv{max_teacher_variance:g}"
    if int(student_pretraining_steps) > 0:
        base_name = f"{base_name}_spts{int(student_pretraining_steps)}"
    if grpo_like:
        base_name = f"{base_name}_grpo"
    base = parent / base_name
    base.mkdir(parents=True, exist_ok=True)
    video_dir = base / "videos"
    video_dir.mkdir(exist_ok=True)
    losses_pdf = base / "losses.pdf"
    return base, video_dir, losses_pdf


_BC_META_KEYS = frozenset(
    {"alignment_ratio", "replan_env_step", "teacher_chunk_var_mean"}
)


def _example_alignment_ratio(ex: dict) -> float:
    v = ex.get("alignment_ratio")
    if v is None:
        return float("nan")
    return float(np.asarray(v, dtype=np.float64).reshape(-1)[0])


def _example_env_step(ex: dict) -> int | None:
    """Env step at replan for this distillation row (see ttt distillation metadata)."""
    v = ex.get("replan_env_step")
    if v is None:
        return None
    return int(np.asarray(v, dtype=np.int64).reshape(-1)[0])


def _example_teacher_chunk_var_mean(ex: dict) -> float:
    """Mean_{h,d} Var_{teacher samples}(chunk[h,d]); missing key -> 0 (legacy rows)."""
    v = ex.get("teacher_chunk_var_mean")
    if v is None:
        return 0.0
    return float(np.asarray(v, dtype=np.float64).reshape(-1)[0])


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


def _filter_examples_by_max_teacher_variance(
    examples: list[dict],
    *,
    max_teacher_variance: float | None,
) -> tuple[list[dict], int, int]:
    """Drop samples with teacher chunk variance (mean over H,D of var across teacher samples) > cap."""
    if max_teacher_variance is None:
        return list(examples), len(examples), 0
    kept: list[dict] = []
    dropped = 0
    cap = float(max_teacher_variance)
    for ex in examples:
        if _example_teacher_chunk_var_mean(ex) <= cap:
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


def _plot_distillation_iter_rollout_metrics_pdf(
    *,
    iter_idx: int,
    task_id: int,
    episode_metrics: list[dict[str, Any]] | None,
    pdf_path: Path,
) -> None:
    """One PDF with four pages: teacher alignment, teacher–student L2, student variance, teacher variance."""
    if not episode_metrics:
        return
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    n_ep = len(episode_metrics)
    ncols = min(3, max(1, n_ep))
    nrows = (n_ep + ncols - 1) // ncols

    def _hide_unused(axes: Any, used: int) -> None:
        for j in range(used, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].set_visible(False)

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
        for i, ep in enumerate(episode_metrics):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            series = ep.get("teacher_alignment_ratio_by_step") or []
            if series:
                ax.plot([p[0] for p in series], [p[1] for p in series], "C0.-", lw=1.2, ms=4)
            ax.set_xlabel("Environment step")
            ax.set_ylabel("Teacher alignment ratio")
            ax.set_title(f"Episode {i + 1}")
            ax.grid(True, alpha=0.3)
        _hide_unused(axes, n_ep)
        fig.suptitle(
            f"Outer iter {iter_idx} — teacher alignment ratio (task {task_id})",
            fontsize=11,
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
        for i, ep in enumerate(episode_metrics):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            series = ep.get("teacher_l2_by_env_step") or []
            if series:
                ax.plot([p[0] for p in series], [p[1] for p in series], "C1.-", lw=1.2, ms=4)
            ax.set_xlabel("Environment step")
            ax.set_ylabel("L2 dist (teacher chunk vs student chunk)")
            ax.set_title(f"Episode {i + 1}")
            ax.grid(True, alpha=0.3)
        _hide_unused(axes, n_ep)
        fig.suptitle(
            f"Outer iter {iter_idx} — teacher vs student action chunk L2 (task {task_id})",
            fontsize=11,
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
        for i, ep in enumerate(episode_metrics):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            series = ep.get("group_sampling_trace") or []
            if series:
                ax.plot(
                    [x["env_step"] for x in series],
                    [float(x["chunk_var_mean"]) for x in series],
                    "C2.-",
                    lw=1.2,
                    ms=4,
                )
            ax.set_xlabel("Environment step")
            ax.set_ylabel("Student chunk variance\n(mean over action dims, var over samples)")
            ax.set_title(f"Episode {i + 1}")
            ax.grid(True, alpha=0.3)
        _hide_unused(axes, n_ep)
        fig.suptitle(
            f"Outer iter {iter_idx} — student action-chunk variance (task {task_id})",
            fontsize=11,
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
        for i, ep in enumerate(episode_metrics):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            series = ep.get("teacher_group_sampling_trace") or []
            if series:
                ax.plot(
                    [x["env_step"] for x in series],
                    [float(x["chunk_var_mean"]) for x in series],
                    "C3.-",
                    lw=1.2,
                    ms=4,
                )
            ax.set_xlabel("Environment step")
            ax.set_ylabel("Teacher chunk variance\n(mean over action dims, var over samples)")
            ax.set_title(f"Episode {i + 1}")
            ax.grid(True, alpha=0.3)
        _hide_unused(axes, n_ep)
        fig.suptitle(
            f"Outer iter {iter_idx} — teacher action-chunk variance (task {task_id})",
            fontsize=11,
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


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
    student_pretrain_losses: list[float] | None = None,
    student_pretrain_eval: list[dict] | None = None,
    student_pretraining_total_steps: int = 0,
    student_pretraining_eval_episodes: int = 10,
) -> None:
    """Multi-page PDF: (1) teacher + optional student SFT prep + BC + success curves, (2) one panel per BC phase."""
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
        spt = student_pretrain_losses or []
        if spt:
            off = len(teacher_losses)
            ax0.plot(
                [off + i for i in range(len(spt))],
                spt,
                "-",
                color="darkorange",
                linewidth=1.3,
                label="Student SFT prep (single-episode demo)",
            )
        ax0.set_ylabel("Loss")
        ax0.set_xlabel("Gradient step (teacher, then SFT prep, then student BC in next panel)")
        ax0.set_title("Teacher + optional student SFT preparation")
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

        spe = student_pretrain_eval or []
        xs_p: list[int] = []
        iters: list[int] = []
        piters: list[int] = []
        if distill_eval_by_iter or periodic_eval_10ep_by_iter or spe:
            if spe and student_pretraining_total_steps > 0:
                xs_p = [
                    int(e["step"]) - student_pretraining_total_steps - 1 for e in spe
                ]
                ys_p = [float(e["success_rate"]) * 100.0 for e in spe]
                ne = int(student_pretraining_eval_episodes)
                ax2.plot(
                    xs_p,
                    ys_p,
                    "D-",
                    color="darkorange",
                    markersize=6,
                    label=(
                        f"SFT prep ({ne}-ep eval, x = step − total − 1)"
                    ),
                )
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
            ax2.set_xlabel(
                "Distillation outer iter (≥0); SFT prep points use negative x (see legend)"
            )
            ax2.set_title(
                "Student success rate: SFT prep eval, rollout distill, periodic 10-ep"
            )
            ax2.set_ylim(-5, 105)
            all_x = xs_p + iters + piters
            if all_x:
                ax2.set_xlim(min(all_x) - 2, max(all_x) + 2)
            ax2.legend(loc="upper right", fontsize=7)
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
        writer.writerow(["student_action_merge", results.get("student_action_merge")])
        writer.writerow(["group_size", results.get("group_size")])
        writer.writerow(["teacher_group_size", results.get("teacher_group_size")])
        writer.writerow(["temporal_decay", results.get("temporal_decay")])
        writer.writerow(["l1_bc_loss", results.get("l1_bc_loss")])
        writer.writerow(["kl_lambda", results.get("kl_lambda")])
        writer.writerow(["max_teacher_variance", results.get("max_teacher_variance")])
        writer.writerow(["grpo_like", results.get("grpo_like")])
        writer.writerow(["student_pretraining_steps", results.get("student_pretraining_steps")])
        writer.writerow(["student_pretraining_lr", results.get("student_pretraining_lr")])
        writer.writerow(
            ["student_pretraining_eval_interval", results.get("student_pretraining_eval_interval")]
        )
        writer.writerow(
            ["student_pretraining_eval_episodes", results.get("student_pretraining_eval_episodes")]
        )
        writer.writerow(
            ["student_sft_prep_final_loss", results.get("student_sft_prep_final_loss")]
        )
        for ev in results.get("student_pretrain_eval_by_step", []) or []:
            writer.writerow(
                [
                    f"student_sft_eval_step{ev['step']}_success_rate",
                    ev["success_rate"],
                ]
            )
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
    parser.add_argument(
        "--student_action_merge",
        type=float,
        default=0.0,
        help=(
            "Distillation BC targets: (1-α)*teacher_action_chunk + α*student_action_chunk per replan, "
            "α in [0,1]. 0 (default) is pure teacher; 1 is pure student."
        ),
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=1,
        help=(
            "Student rollout: G independent action-chunk samples per replan (same obs); "
            "first sample drives the env. G>1 populates student variance in distillation_rollout_metrics.pdf."
        ),
    )
    parser.add_argument(
        "--teacher_group_size",
        type=int,
        default=1,
        help=(
            "Distillation rollout: T independent teacher noise draws per replan. T==1 uses the student's "
            "first noise (legacy). T>1 measures teacher chunk spread; first teacher sample is still the "
            "BC target. Teacher variance is plotted in distillation_rollout_metrics.pdf."
        ),
    )
    parser.add_argument(
        "--temporal_decay",
        type=float,
        default=1.0,
        help=(
            "Per-replan BC sample weight = temporal_decay ** (env step at replan). "
            "1.0 (default) = uniform. Use <1 to down-weight later timesteps."
        ),
    )
    parser.add_argument(
        "--l1_bc_loss",
        action="store_true",
        help="Student BC only: train diffusion residual with L1 (|v-u|) instead of L2 MSE.",
    )
    parser.add_argument(
        "--kl_lambda",
        type=float,
        default=0.0,
        help=(
            "Student BC only (Pi0): add kl_lambda * MSE(v_t, stop_grad(v_t^ref)) vs a frozen snapshot "
            "of the student at the start of that BC segment; same noise/timestep as teacher loss."
        ),
    )
    parser.add_argument(
        "--max_teacher_variance",
        type=float,
        default=None,
        help=(
            "If set, drop BC rows where teacher action-chunk variance exceeds this value. "
            "Variance matches distillation_rollout_metrics / teacher trace: mean over (horizon, dim) of "
            "variance across teacher_group_size samples. Requires teacher_group_size>1 for non-trivial spread; "
            "otherwise metadata variance is 0. Omit to disable."
        ),
    )
    parser.add_argument(
        "--student_pretraining_steps",
        type=int,
        default=100,
        help=(
            "Before OPD: offline SFT on a fresh student copy using a single-episode demo of the same task "
            "(independent of --single_episode used for the teacher). 0 disables this stage."
        ),
    )
    parser.add_argument(
        "--student_pretraining_lr",
        type=float,
        default=None,
        help="LR for student SFT prep; default: same as --teacher_lr.",
    )
    parser.add_argument(
        "--student_pretraining_eval_interval",
        type=int,
        default=50,
        help="During student SFT prep: run this many grad steps between 10-rollout evals (and always after the last step).",
    )
    parser.add_argument(
        "--student_pretraining_eval_episodes",
        type=int,
        default=10,
        help="Episodes per eval during student SFT prep.",
    )
    parser.add_argument(
        "--grpo_like",
        action="store_true",
        help=(
            "GRPO-style student update: per env step sample G=--group_size student chunks, "
            "R_i=-||a_i-a_teacher||_2, A_hat_i = z-score of R within the group; "
            "BC objective mean_i (A_hat_i * diffusion denoise loss on (s,a_i)). Requires group_size>=2."
        ),
    )
    args = parser.parse_args()
    if args.alignment_ratio_threshold is not None and args.align_min is not None:
        parser.error("use only one of --alignment_ratio_threshold and --align_min")
    sam = float(args.student_action_merge)
    if not (0.0 <= sam <= 1.0):
        parser.error("--student_action_merge must be in [0, 1]")
    group_size = max(1, int(args.group_size))
    teacher_group_size = max(1, int(args.teacher_group_size))
    if args.grpo_like and group_size < 2:
        parser.error("--grpo_like requires --group_size >= 2 (per-step group for advantages)")
    if args.grpo_like and abs(sam) > 1e-12:
        parser.error("--grpo_like is incompatible with --student_action_merge != 0 (use pure student chunks a_i)")
    temporal_decay = float(args.temporal_decay)
    if temporal_decay < 0:
        parser.error("--temporal_decay must be >= 0")
    kl_lambda = float(args.kl_lambda)
    if kl_lambda < 0.0:
        parser.error("--kl_lambda must be >= 0")
    max_teacher_var = args.max_teacher_variance
    if max_teacher_var is not None and max_teacher_var < 0.0:
        parser.error("--max_teacher_variance must be >= 0 when set")

    task_id = args.task
    seed = args.seed
    batch_size = args.batch_size
    teacher_steps = args.teacher_steps
    teacher_lr = args.teacher_lr
    student_pretraining_steps = max(0, int(args.student_pretraining_steps))
    spt_lr = (
        float(args.student_pretraining_lr)
        if args.student_pretraining_lr is not None
        else teacher_lr
    )
    if student_pretraining_steps > 0 and spt_lr <= 0.0:
        parser.error("student SFT prep requires a positive LR (use --student_pretraining_lr or positive --teacher_lr)")
    if args.student_pretraining_lr is not None and float(args.student_pretraining_lr) <= 0.0:
        parser.error("--student_pretraining_lr must be > 0 when set")
    spt_eval_interval = max(1, int(args.student_pretraining_eval_interval))
    spt_eval_episodes = max(1, int(args.student_pretraining_eval_episodes))
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
        student_action_merge=sam,
        group_size=group_size,
        teacher_group_size=teacher_group_size,
        temporal_decay=temporal_decay,
        l1_bc_loss=bool(args.l1_bc_loss),
        kl_lambda=kl_lambda,
        max_teacher_variance=max_teacher_var,
        student_pretraining_steps=student_pretraining_steps,
        grpo_like=bool(args.grpo_like),
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

    student_pretrain_losses: list[float] = []
    student_pretrain_eval: list[dict] = []

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
            "(student alignment in BC metadata; rollout plots: distillation_rollout_metrics.pdf uses teacher alignment)"
        )
    elif align_min is not None:
        print(
            f"Alignment filtering (align_min): keep BC samples with alignment_ratio >= {align_min} "
            "(student alignment in BC metadata)"
        )
    if max_teacher_var is not None:
        print(
            f"Teacher-variance filtering: keep BC samples with teacher_chunk_var_mean <= {max_teacher_var} "
            "(mean_{h,d} Var across teacher_group_size samples; see --teacher_group_size)"
        )

    def _flush_losses_pdf(
        where: str,
        *,
        running_iter: int | None = None,
        running_rollout_sr: float | None = None,
    ) -> None:
        if running_iter is not None and running_rollout_sr is not None:
            distill_plot = list(distill_eval_by_iter) + [
                {"iter": int(running_iter), "success_rate": float(running_rollout_sr)}
            ]
        else:
            distill_plot = list(distill_eval_by_iter) if distill_eval_by_iter else None
        _plot_losses_pdf(
            teacher_losses=teacher_losses,
            bc_losses=all_bc_losses,
            bc_step_offsets=bc_offsets,
            bc_loss_runs=bc_loss_runs,
            bc_run_iters=bc_run_iters,
            distill_eval_by_iter=distill_plot,
            periodic_eval_10ep_by_iter=(
                periodic_eval_10ep_by_iter if args.full_experiment else None
            ),
            pdf_path=losses_pdf,
            student_pretrain_losses=student_pretrain_losses if student_pretrain_losses else None,
            student_pretrain_eval=student_pretrain_eval if student_pretrain_eval else None,
            student_pretraining_total_steps=student_pretraining_steps,
            student_pretraining_eval_episodes=spt_eval_episodes,
        )
        print(f"  Updated {losses_pdf} ({where})")

    # Student after teacher eval; SFT prep may call _flush_losses_pdf.
    print("\n" + "=" * 50)
    print("Student: fresh copy of base model (not teacher)")
    print("=" * 50)
    student_model = copy_model(base_model, config)
    student_model.eval()

    if student_pretraining_steps > 0:
        print("\n" + "=" * 50)
        print(
            f"Student SFT preparation: {student_pretraining_steps} BC steps on a single-episode "
            f"demo of task {task_id} (separate from teacher data; teacher used --single_episode={args.single_episode}); "
            f"lr={spt_lr}; eval every {spt_eval_interval} steps, {spt_eval_episodes} episodes per eval"
        )
        print("=" * 50)
        with override_create_torch_dataset(
            repo_id=repo_id,
            task_id=task_id,
            mirror_data=True,
            single_episode=True,
            augment=False,
        ):
            spt_loader = _data_loader.create_data_loader(
                config,
                sharding=None,
                shuffle=True,
            )
        resume_spt_ts = None
        resume_spt_losses: list[float] | None = None
        spt_done = 0
        spt_prep_video = video_dir / "student_sft_prep_eval"
        spt_prep_video.mkdir(parents=True, exist_ok=True)
        while spt_done < student_pretraining_steps:
            chunk = min(spt_eval_interval, student_pretraining_steps - spt_done)
            student_model, student_pretrain_losses, resume_spt_ts = train_model_on_fly(
                model=student_model,
                training_data_loader=spt_loader,
                config=config,
                learning_rate=spt_lr,
                num_steps=chunk,
                warmup_steps=0,
                weight_decay=0.0,
                log_interval=max(1, chunk // 5),
                seed=seed + 333_000 + spt_done,
                show_progress_bar=True,
                donate_buffers=False,
                resume_train_state=resume_spt_ts,
                resume_losses=resume_spt_losses,
            )
            resume_spt_losses = student_pretrain_losses
            spt_done += chunk
            student_model.eval()
            ev_dir = spt_prep_video / f"step_{spt_done:04d}"
            ev_dir.mkdir(parents=True, exist_ok=True)
            spt_pol = create_policy(
                student_model,
                config,
                CHECKPOINT_DIR,
                rng_seed=seed + spt_done + 202_020,
            )
            sr, _ = run_evaluation(
                policy=spt_pol,
                train_config=config,
                num_trials=spt_eval_episodes,
                task_suite_name=TASK_SUITE_NAME,
                task_id=task_id,
                save_video=args.save_video,
                video_out_path=str(ev_dir),
                seed=seed + spt_done + 101_010,
                teacher_policy=None,
                show_progress_bar=True,
            )
            student_pretrain_eval.append({"step": spt_done, "success_rate": float(sr)})
            print(
                f"  SFT prep eval at grad step {spt_done}/{student_pretraining_steps}: "
                f"{sr * 100:.1f}% "
                f"({int(round(sr * spt_eval_episodes))}/{spt_eval_episodes} episodes)"
            )
            del spt_pol
            _flush_losses_pdf(f"student SFT prep after step {spt_done}")
        del spt_loader
        gc.collect()

    del base_model
    gc.collect()

    _flush_losses_pdf("after teacher + student SFT prep (if any); start distillation")

    for it in range(max_iters):
        iter_video = video_dir / f"iter_{it:03d}"
        iter_video.mkdir(parents=True, exist_ok=True)
        episode_examples: list[dict] = []

        student_policy = create_policy(student_model, config, CHECKPOINT_DIR, rng_seed=seed + it * 100_003)

        print(
            f"\n--- Distillation iter {it + 1}/{max_iters}: "
            f"rollout ({rollout_episodes} ep{'s' if rollout_episodes != 1 else ''}) ---"
        )

        def _after_rollout_ep(ep_idx: int, n_ep: int, n_succ: int) -> None:
            sr_run = (n_succ / n_ep) if n_ep else 0.0
            _flush_losses_pdf(
                f"outer iter {it}, after rollout episode {ep_idx + 1}/{rollout_episodes} "
                f"(running rollout success rate {sr_run:.2f})",
                running_iter=it,
                running_rollout_sr=sr_run,
            )

        success_rate, episode_rollout_metrics = run_evaluation(
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
            student_action_merge=sam,
            group_size=group_size,
            teacher_group_size=teacher_group_size,
            temporal_decay=temporal_decay,
            write_auxiliary_rollout_pdfs=False,
            after_each_rollout_episode=_after_rollout_ep,
            grpo_like=bool(args.grpo_like),
        )
        if episode_rollout_metrics:
            _plot_distillation_iter_rollout_metrics_pdf(
                iter_idx=it,
                task_id=task_id,
                episode_metrics=episode_rollout_metrics,
                pdf_path=iter_video / "distillation_rollout_metrics.pdf",
            )
        # Stop only when every rollout episode succeeded (mean rate == 1.0).
        ep_success = success_rate >= 1.0 - 1e-6
        n_new = len(episode_examples)
        ratios = [_example_alignment_ratio(e) for e in episode_examples]
        filtered_ep, n_kept_al, n_drop_al = _filter_examples_by_alignment(
            episode_examples,
            alignment_ratio_threshold=align_th,
            align_min=align_min,
        )
        filtered_ep, n_kept_tv, n_drop_tv = _filter_examples_by_max_teacher_variance(
            filtered_ep,
            max_teacher_variance=max_teacher_var,
        )
        stripped_for_bc = [_strip_bc_metadata(e) for e in filtered_ep]
        if args.cumulative_buffer:
            bc_buffer.extend(stripped_for_bc)
        else:
            bc_buffer = list(stripped_for_bc)

        distill_eval_by_iter.append({"iter": it, "success_rate": float(success_rate)})
        buf_note = "cumulative" if args.cumulative_buffer else "last-episode only"
        al_part = (
            f", align kept {n_kept_al}/{n_new} (drop {n_drop_al})"
            if (align_th is not None or align_min is not None)
            else ""
        )
        tv_part = (
            f", teacher_var kept {n_kept_tv}/{n_kept_al if (align_th is not None or align_min is not None) else n_new} "
            f"(drop {n_drop_tv})"
            if max_teacher_var is not None
            else ""
        )
        print(
            f"  Iter {it}: success_rate={success_rate:.2f}, new_samples={n_new}{al_part}{tv_part}, "
            f"BC data ({buf_note})={len(bc_buffer)}"
        )

        if ep_success:
            final_success = True
            if not args.full_experiment:
                stopped_reason = "rollout_success"
                break

        if not bc_buffer:
            if n_new > 0 and (
                align_th is not None
                or align_min is not None
                or max_teacher_var is not None
            ):
                print(
                    "  No BC examples left after alignment / teacher-variance filter; skipping train step."
                )
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
            _flush_losses_pdf(f"outer iter {it}, after periodic eval (no BC this iter)")
            continue

        data_cfg = config.data.create(config.assets_dirs, config.model)
        bc_bs = min(batch_size, len(bc_buffer))
        bc_loader = NeighborsDataLoader(bc_buffer, bc_bs, data_cfg)

        _bc_note = " (GRPO-weighted denoise loss)" if args.grpo_like else ""
        print(
            f"  BC train: {bc_steps} steps, lr={bc_lr}, examples={len(bc_buffer)}, batch={bc_bs}{_bc_note}"
        )
        bc_loss_chunk: list[float] = []

        def _on_bc_step(_step: int, loss_val: float) -> None:
            bc_loss_chunk.append(loss_val)

        ref_for_kl = None
        if kl_lambda > 0.0:
            ref_for_kl = copy_model(student_model, config)
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
            l1_loss=bool(args.l1_bc_loss),
            kl_lambda=kl_lambda,
            ref_model_for_kl=ref_for_kl,
            grpo_like=bool(args.grpo_like),
        )
        del ref_for_kl
        del bc_loader
        student_model.eval()
        all_bc_losses.extend(bc_loss_chunk)
        bc_offsets.append(len(all_bc_losses))
        bc_loss_runs.append(list(bc_loss_chunk))
        bc_run_iters.append(it)
        gc.collect()
        _flush_losses_pdf(f"outer iter {it}, after student BC")

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
            _flush_losses_pdf(f"outer iter {it}, after periodic eval (full_experiment)")

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
        "student_action_merge": sam,
        "group_size": group_size,
        "teacher_group_size": teacher_group_size,
        "temporal_decay": temporal_decay,
        "l1_bc_loss": bool(args.l1_bc_loss),
        "kl_lambda": kl_lambda,
        "max_teacher_variance": max_teacher_var,
        "grpo_like": bool(args.grpo_like),
        "student_pretraining_steps": student_pretraining_steps,
        "student_pretraining_lr": (float(spt_lr) if student_pretraining_steps > 0 else None),
        "student_pretraining_eval_interval": spt_eval_interval,
        "student_pretraining_eval_episodes": spt_eval_episodes,
        "student_pretrain_eval_by_step": list(student_pretrain_eval),
        "student_sft_prep_final_loss": (
            float(student_pretrain_losses[-1]) if student_pretrain_losses else None
        ),
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
        student_pretrain_losses=student_pretrain_losses if student_pretrain_losses else None,
        student_pretrain_eval=student_pretrain_eval if student_pretrain_eval else None,
        student_pretraining_total_steps=student_pretraining_steps,
        student_pretraining_eval_episodes=spt_eval_episodes,
    )
    print(
        f"\nFinal write of losses PDF to {losses_pdf} "
        f"(also updated during SFT prep, each rollout episode, after BC, and after teacher eval; "
        f"page 1: teacher + optional SFT prep + BC + success curves; page 2+: per-BC-phase)"
    )
    print(
        f"Distillation rollout metrics (4-page PDF per iter): "
        f"{video_dir}/iter_*/distillation_rollout_metrics.pdf"
    )
    print(f"Saved results to {run_dir / 'results.csv'} (includes teacher_eval_success_rate)")


if __name__ == "__main__":
    main()
