"""PDF plotting for on-policy distillation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from meta_libero.src.on_policy_distillation.bc_data import kept_by_alignment_policy
from meta_libero.src.on_policy_distillation.constants import FULL_EXPERIMENT_EVAL_INTERVAL


def plot_alignment_ratios_iter_pdf(
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
            if kept_by_alignment_policy(
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


def plot_distillation_iter_rollout_metrics_pdf(
    *,
    iter_idx: int,
    task_id: int,
    episode_metrics: list[dict[str, Any]] | None,
    pdf_path: Path,
) -> None:
    """One PDF: teacher alignment, L2, variances, GRPO rewards, GRPO advantages, mean/std weight."""
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

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
        for i, ep in enumerate(episode_metrics):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            series = ep.get("group_sampling_trace") or []
            xs: list[float] = []
            ys: list[float] = []
            cs: list[int] = []
            for row in series:
                t = int(row["env_step"])
                dists = row.get("dists") or []
                g = len(dists)
                if g == 0:
                    continue
                for j, d in enumerate(dists):
                    xs.append(t + 0.12 * (j - (g - 1) / 2.0))
                    ys.append(-float(d))
                    cs.append(j % 10)
            if xs:
                ax.scatter(
                    xs,
                    ys,
                    c=cs,
                    cmap="tab10",
                    vmin=0,
                    vmax=9,
                    s=26,
                    alpha=0.85,
                    edgecolors="0.35",
                    linewidths=0.35,
                )
            ax.set_xlabel("Environment step")
            ax.set_ylabel(r"GRPO reward $R_i = -\|a_i - a_{\mathrm{teach}}\|_2$")
            ax.set_title(f"Episode {i + 1}")
            ax.grid(True, alpha=0.3)
        _hide_unused(axes, n_ep)
        fig.suptitle(
            f"Outer iter {iter_idx} — GRPO rewards (color = sample index within group; task {task_id})",
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
            xs_a: list[float] = []
            ys_a: list[float] = []
            cs_a: list[int] = []
            for row in series:
                t = int(row["env_step"])
                advs = row.get("grpo_advantages") or []
                g = len(advs)
                if g == 0:
                    continue
                for j, a in enumerate(advs):
                    xs_a.append(t + 0.12 * (j - (g - 1) / 2.0))
                    ys_a.append(float(a))
                    cs_a.append(j % 10)
            if xs_a:
                ax.scatter(
                    xs_a,
                    ys_a,
                    c=cs_a,
                    cmap="tab10",
                    vmin=0,
                    vmax=9,
                    s=26,
                    alpha=0.85,
                    edgecolors="0.35",
                    linewidths=0.35,
                )
            ax.set_xlabel("Environment step")
            ax.set_ylabel(
                r"$\hat{A}_i$ (z-score of $R_i$ within group; $R_i=-\|a_i-a_{\mathrm{teach}}\|_2$)"
            )
            ax.set_title(f"Episode {i + 1}")
            ax.grid(True, alpha=0.3)
        _hide_unused(axes, n_ep)
        fig.suptitle(
            f"Outer iter {iter_idx} — GRPO advantages (color = sample index; task {task_id})",
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
            xs_w: list[int] = []
            ys_w: list[float] = []
            for row in series:
                wv = row.get("grpo_mean_std_weight")
                if wv is not None and np.isfinite(float(wv)):
                    xs_w.append(int(row["env_step"]))
                    ys_w.append(float(wv))
            if xs_w:
                ax.plot(xs_w, ys_w, "C4.-", lw=1.2, ms=4)
            ax.set_xlabel("Environment step")
            ax.set_ylabel(
                r"$w_t=\frac{\mathrm{mean}\,\|d_i\|}{\mathrm{std}\,\|d_i\|+\epsilon}$"
                r" ($d_i=a_i-a_{\mathrm{teach}}$)"
            )
            ax.set_title(f"Episode {i + 1}")
            ax.grid(True, alpha=0.3)
        _hide_unused(axes, n_ep)
        fig.suptitle(
            f"Outer iter {iter_idx} — GRPO mean/std group weight (task {task_id})",
            fontsize=11,
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def plot_losses_pdf(
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
    bc_param_delta_l2_by_iter: list[float] | None = None,
    bc_param_delta_iters: list[int] | None = None,
) -> None:
    """Multi-page PDF: (1) teacher + optional SFT prep + BC + success + ||Δθ|| after each BC phase, (2) one panel per BC phase."""
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=False)
        ax0, ax1, ax2, ax3 = axes[0], axes[1], axes[2], axes[3]
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

        bpd = bc_param_delta_l2_by_iter or []
        bpi = bc_param_delta_iters or []
        if bpd and bpi and len(bpd) == len(bpi):
            ax3.plot(
                bpi,
                bpd,
                "s-",
                color="teal",
                linewidth=1.2,
                markersize=5,
                label=r"$\|\Delta\theta\|_2$ after BC phase",
            )
            ax3.set_ylabel(r"$\|\Delta\theta\|_2$ (trainable)")
            ax3.set_xlabel("Distillation outer iter (BC phase index)")
            ax3.set_title(
                "Student parameter change per BC phase (L2 norm of trainable weights before vs after BC)"
            )
            ax3.legend(loc="upper right", fontsize=8)
            ax3.grid(True, alpha=0.3)
            if len(bpi) == 1:
                ax3.set_xlim(bpi[0] - 0.5, bpi[0] + 0.5)
        else:
            ax3.set_visible(False)

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
