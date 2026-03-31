## Sequential two-task on-policy distillation.
#
# Phase 1 (task): teacher from base -> FT on task1; student from base -> distillation on task1.
# Phase 2 (task2): new teacher from base -> FT on task2; same student weights (no reset) ->
# distillation on task2. Each outer iteration on task2: rollout on task2 for BC, then separate
# 10-episode evals on task1 and task2 (no teacher). Optional --phase2_self_replay pairs each OPD row
# (task2 + teacher targets) with the same observation under the task1 prompt and reference-student
# pseudo-actions in one interleaved BC batch (no separate offline dataset phase).
# Videos (if --save_video): only phase-1 periodic 10-ep eval and phase-2 dual 10-ep eval; not rollouts/teachers/final.
#
# LoRA / action-expert-only: same as augment_finetune_experiment.py — pass --lora to load
# pi05_libero_lora and fine-tune adapters; or --action_expert_only (mutually exclusive with --lora).
# Run dirs: .../on_policy_distillation_two_task[_single][_lora|_action_expert]/task{t1}_task{t2}_seed{s}/
#   tlr{lr}_ts1_{steps}_ts2_{steps2}_bc_lr{lr}_steps{bc}_mi1_{iters}_mi2_{iters2}[_align...][_rolloutEpN]
#
# Usage:
#   python on_policy_distillation_two_task.py --task 0 --task2 1 --max_iters 25 --max_iters_task2 25
#   python on_policy_distillation_two_task.py --task 0 --task2 1 --lora

import importlib.util
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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import openpi.transforms as op_transforms
from openpi.models import tokenizer as _pi_tokenizer  # type: ignore

from openpi.training import data_loader as _data_loader  # type: ignore

from meta_libero.src.dataset import (  # type: ignore
    LIBERO_90_TASK_IDS_PROMPTS,
    make_pseudo_label_inference_fn,
    override_create_torch_dataset,
)
from meta_libero.src.ttt import (  # type: ignore
    PAIR_STREAM_INTERLEAVED,
    NeighborsDataLoader,
    PairedOnPolicySelfReplayDataLoader,
    _pad_action_chunk_for_model,
    copy_model,
    create_policy,
    load_pi05_libero_model,
    run_evaluation,
    train_model_on_fly,
)

# Match on_policy_distillation.py (do not diverge for CHECKPOINT_DIR / task suite).
TASK_SUITE_NAME = "libero_90"
STUDENT_FINAL_EVAL_EPISODES = 10
CHECKPOINT_DIR = os.getenv(
    "OPENPI_CHECKPOINT_DIR",
    str(Path.home() / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero"),
)

# Lazy-load single-task module for alignment helpers only; avoids import/exec work before teacher FT+eval.
_OP_PATH = Path(__file__).resolve().parent / "on_policy_distillation.py"
_opd = None


def _get_opd():
    global _opd
    if _opd is None:
        spec = importlib.util.spec_from_file_location("_opd_single", _OP_PATH)
        assert spec and spec.loader
        _opd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_opd)
    return _opd


def _phase2_opd_paired_task1_pseudo_examples(
    opd_stripped: list[dict],
    ref_policy: Any,
    task1_prompt: str,
    *,
    model_action_horizon: int,
    model_action_dim: int,
) -> list[dict]:
    """For each OPD (task2) BC row, build a paired row: same images/state, task1 language, actions = ref policy.

    BC batches collate + ``to_jax`` every leaf; raw string ``prompt`` is invalid (Unicode dtype). We store
    ``tokenized_prompt`` / ``tokenized_prompt_mask`` like ``TokenizePrompt`` with ``state=None`` (pi05_libero:
    ``discrete_state_input=False``).
    """
    if task1_prompt is None or not str(task1_prompt).strip():
        raise ValueError("task1_prompt must be a non-empty string")
    infer = make_pseudo_label_inference_fn(ref_policy)
    max_tl = int(ref_policy._model.max_token_len)
    pg_tok = _pi_tokenizer.PaligemmaTokenizer(max_tl)
    out: list[dict] = []
    for ex in opd_stripped:
        d = dict(ex)
        d["prompt"] = task1_prompt
        act = np.asarray(infer(d), dtype=np.float32)
        act = _pad_action_chunk_for_model(
            act,
            action_horizon=model_action_horizon,
            action_dim=model_action_dim,
        )
        aug = dict(ex)
        aug.pop("prompt", None)
        toks, tmask = pg_tok.tokenize(str(task1_prompt).strip(), state=None)
        aug["tokenized_prompt"] = np.asarray(toks, dtype=np.int32)
        aug["tokenized_prompt_mask"] = np.asarray(tmask, dtype=np.bool_)
        aug["actions"] = act
        if aug.get("state") is not None:
            st = np.asarray(aug["state"], dtype=np.float32)
            if st.shape[-1] > model_action_dim:
                st = st[..., :model_action_dim]
            else:
                st = op_transforms.pad_to_dim(st, model_action_dim, axis=-1)
            aug["state"] = st
        if ex.get("distill_sample_weight") is not None:
            aug["distill_sample_weight"] = ex["distill_sample_weight"]
        out.append(aug)
    return out


DUAL_EVAL_EPISODES = 10
# Match on_policy_distillation.py --full_experiment
FULL_EXPERIMENT_EVAL_INTERVAL = 5
FULL_EXPERIMENT_EVAL_EPISODES = 10


def _setup_logging() -> None:
    """Same as on_policy_distillation._setup_logging (duplicated here so we do not import _opd at startup)."""

    class VersionWarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "is in 2.0 format" not in record.getMessage()

    logging.getLogger().addFilter(VersionWarningFilter())
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logging.getLogger("OpenGL").setLevel(logging.ERROR)


def _gc_after_bc_train() -> None:
    """After BC only: two-task needs this; single-task does not call clear_caches between rollouts."""
    import jax as _jax

    gc.collect()
    _jax.clear_caches()


def _results_root() -> Path:
    return Path(os.getenv("META_LIBERO_RESULTS_DIR", "meta_libero/results"))


def _ensure_run_dir_two_task(
    task1: int,
    task2: int,
    seed: int,
    teacher_lr: float,
    teacher_steps: int,
    teacher_steps_task2: int,
    bc_lr: float,
    bc_steps: int,
    max_iters: int,
    max_iters_task2: int,
    single_episode: bool = False,
    lora: bool = False,
    action_expert_only: bool = False,
    alignment_ratio_threshold: float | None = None,
    align_min: float | None = None,
    rollout_episodes: int = 1,
    full_experiment: bool = False,
    phase2_eval_interval: int = 5,
    student_action_merge: float = 0.0,
    phase2_self_replay: bool = False,
    temporal_decay: float = 1.0,
    l1_bc_loss: bool = False,
    kl_lambda: float = 0.0,
    group_size: int = 1,
    teacher_group_size: int = 1,
) -> tuple[Path, Path, Path]:
    folder_base = "on_policy_distillation_two_task_single" if single_episode else "on_policy_distillation_two_task"
    if lora:
        folder_base = f"{folder_base}_lora"
    elif action_expert_only:
        folder_base = f"{folder_base}_action_expert"
    parent = _results_root() / folder_base / f"task{task1}_task{task2}_seed{seed}"
    # Unambiguous tokens (avoid "mi5_mi225" reading as mi=225; use mi1_ / mi2_ prefixes).
    base_name = (
        f"tlr{teacher_lr}_ts1_{teacher_steps}_ts2_{teacher_steps_task2}_"
        f"bc_lr{bc_lr}_steps{bc_steps}_mi1_{max_iters}_mi2_{max_iters_task2}"
    )
    if rollout_episodes != 1:
        base_name = f"{base_name}_rolloutEp{rollout_episodes}"
    if alignment_ratio_threshold is not None:
        base_name = f"{base_name}_align{alignment_ratio_threshold}"
    elif align_min is not None:
        base_name = f"{base_name}_alignmin{align_min}"
    if full_experiment:
        base_name = f"{base_name}_full"
    if phase2_eval_interval != 5:
        base_name = f"{base_name}_p2ev{phase2_eval_interval}"
    if abs(float(student_action_merge)) > 1e-12:
        base_name = f"{base_name}_sam{student_action_merge:g}"
    if phase2_self_replay:
        base_name = f"{base_name}_p2sr"
    if abs(float(temporal_decay) - 1.0) > 1e-12:
        base_name = f"{base_name}_td{temporal_decay:g}"
    if l1_bc_loss:
        base_name = f"{base_name}_l1bc"
    if abs(float(kl_lambda)) > 1e-12:
        base_name = f"{base_name}_kl{kl_lambda:g}"
    if int(group_size) != 1:
        base_name = f"{base_name}_g{int(group_size)}"
    if int(teacher_group_size) != 1:
        base_name = f"{base_name}_tg{int(teacher_group_size)}"
    base = parent / base_name
    base.mkdir(parents=True, exist_ok=True)
    video_dir = base / "videos"
    video_dir.mkdir(exist_ok=True)
    losses_pdf = base / "losses.pdf"
    return base, video_dir, losses_pdf


def _dual_eval_student_10ep(
    *,
    student_model,
    config,
    task1_id: int,
    task2_id: int,
    seed: int,
    outer_iter: int,
    video_dir: Path,
    save_video: bool,
) -> tuple[float, float]:
    """Two separate 10-episode evals (task1 and task2), no teacher."""
    base = video_dir / "task2_dual_eval_10ep" / f"iter_{outer_iter:03d}"
    out1 = base / "eval_task1"
    out2 = base / "eval_task2"
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)
    pol1 = create_policy(
        student_model,
        config,
        CHECKPOINT_DIR,
        rng_seed=seed + outer_iter * 100_003 + 111_111,
    )
    sr1, _ = run_evaluation(
        policy=pol1,
        train_config=config,
        num_trials=DUAL_EVAL_EPISODES,
        task_suite_name=TASK_SUITE_NAME,
        task_id=task1_id,
        save_video=save_video,
        video_out_path=str(out1),
        seed=seed + outer_iter + 222_222,
        teacher_policy=None,
        show_progress_bar=True,
    )
    pol2 = create_policy(
        student_model,
        config,
        CHECKPOINT_DIR,
        rng_seed=seed + outer_iter * 100_003 + 333_333,
    )
    sr2, _ = run_evaluation(
        policy=pol2,
        train_config=config,
        num_trials=DUAL_EVAL_EPISODES,
        task_suite_name=TASK_SUITE_NAME,
        task_id=task2_id,
        save_video=save_video,
        video_out_path=str(out2),
        seed=seed + outer_iter + 444_444,
        teacher_policy=None,
        show_progress_bar=True,
    )
    return float(sr1), float(sr2)


def _periodic_student_eval_10ep(
    *,
    student_model,
    config,
    task_id: int,
    seed: int,
    outer_iter: int,
    video_dir: Path,
    save_video: bool,
    video_subdir: str = "periodic_eval_10ep",
) -> float:
    """10-episode student eval without teacher (same idea as on_policy_distillation --full_experiment)."""
    out = video_dir / video_subdir / f"iter_{outer_iter:03d}"
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


def _plot_two_task_pdf(
    *,
    teacher1_losses: list[float],
    teacher2_losses: list[float],
    teacher_eval_sr1: float,
    teacher_eval_sr2: float | None,
    bc_losses: list[float],
    bc_step_offsets: list[int],
    bc_loss_runs: list[list[float]],
    bc_run_iters: list[int | str],
    phase1_distill: list[dict],
    phase2_metrics: list[dict],
    periodic_eval_10ep_phase1: list[dict] | None,
    phase2_eval_interval: int,
    pdf_path: Path,
) -> None:
    """Page 1: two teachers + BC + phase1 SR + phase2 curves. Page 2: BC grid."""
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=False)
        ax0, ax1, ax2, ax3, ax4 = axes

        if teacher1_losses:
            ax0.plot(range(len(teacher1_losses)), teacher1_losses, "b-", label="Teacher task1", lw=1.5)
        ax0.set_ylabel("Loss")
        ax0.set_xlabel("Gradient step")
        ax0.set_title(
            f"Teacher fine-tune — task 1 (teacher eval SR: {teacher_eval_sr1 * 100:.1f}%)"
        )
        ax0.legend(loc="upper right", fontsize=8)
        ax0.grid(True, alpha=0.3)

        if teacher2_losses:
            ax1.plot(range(len(teacher2_losses)), teacher2_losses, "navy", label="Teacher task2", lw=1.5)
        ax1.set_ylabel("Loss")
        ax1.set_xlabel("Gradient step")
        if teacher_eval_sr2 is not None:
            ax1.set_title(
                f"Teacher fine-tune — task 2 (from base) (teacher eval SR: {teacher_eval_sr2 * 100:.1f}%)"
            )
        else:
            ax1.set_title("Teacher fine-tune — task 2 (from base) — teacher eval pending")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        if bc_losses:
            xs = list(range(len(bc_losses)))
            ax2.plot(xs, bc_losses, "c-", label="Student BC (task1 then task2)", lw=1.0)
            for off in bc_step_offsets[1:]:
                if 0 < off < len(bc_losses):
                    ax2.axvline(off - 0.5, color="0.7", linestyle=":", lw=0.8)
        ax2.set_xlabel("BC gradient step (cumulative)")
        ax2.set_ylabel("Loss")
        ax2.set_title("Student BC loss (continuous student across both tasks)")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        if phase1_distill or periodic_eval_10ep_phase1:
            if phase1_distill:
                iters = [int(x["iter"]) for x in phase1_distill]
                srs = [float(x["success_rate"]) * 100.0 for x in phase1_distill]
                ax3.plot(iters, srs, "g-o", label="Rollout SR (task1 phase)", lw=1.2, ms=4)
            if periodic_eval_10ep_phase1:
                pit = [int(x["iter"]) for x in periodic_eval_10ep_phase1]
                psr = [float(x["success_rate"]) * 100.0 for x in periodic_eval_10ep_phase1]
                ax3.plot(
                    pit,
                    psr,
                    "m-^",
                    label=f"10-ep eval task1 (every {FULL_EXPERIMENT_EVAL_INTERVAL} iters)",
                    lw=1.2,
                    ms=7,
                )
        ax3.set_ylabel("Success %")
        ax3.set_xlabel("Outer iter (task 1)")
        ax3.set_title("Phase 1 — distillation rollout vs periodic 10-ep eval (task 1)")
        ax3.set_ylim(-5, 105)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)

        if phase2_metrics:
            it2 = [int(x["iter"]) for x in phase2_metrics]
            r2 = [float(x["rollout_success_rate_task2"]) * 100.0 for x in phase2_metrics]
            ax4.plot(it2, r2, "g-s", label="Rollout SR (task2, training)", lw=1.2, ms=4)
            ev_rows = [
                x
                for x in phase2_metrics
                if x.get("eval10_task1") is not None and x.get("eval10_task2") is not None
            ]
            if ev_rows:
                it_ev = [int(x["iter"]) for x in ev_rows]
                e1 = [float(x["eval10_task1"]) * 100.0 for x in ev_rows]
                e2 = [float(x["eval10_task2"]) * 100.0 for x in ev_rows]
                ev_lbl = (
                    f"Eval {DUAL_EVAL_EPISODES}ep task1 (every {phase2_eval_interval} iters)"
                    if phase2_eval_interval > 1
                    else f"Eval {DUAL_EVAL_EPISODES}ep task1"
                )
                ev_lbl2 = (
                    f"Eval {DUAL_EVAL_EPISODES}ep task2 (every {phase2_eval_interval} iters)"
                    if phase2_eval_interval > 1
                    else f"Eval {DUAL_EVAL_EPISODES}ep task2"
                )
                ax4.plot(it_ev, e1, "C1-^", label=ev_lbl, lw=1.2, ms=5)
                ax4.plot(it_ev, e2, "C2-d", label=ev_lbl2, lw=1.2, ms=5)
        ax4.set_ylabel("Success %")
        ax4.set_xlabel("Outer iter (task 2)")
        p2_sub = (
            f"dual {DUAL_EVAL_EPISODES}-ep evals every {phase2_eval_interval} outer iters"
            if phase2_eval_interval > 1
            else f"dual {DUAL_EVAL_EPISODES}-ep evals each outer iter"
        )
        ax4.set_title(f"Phase 2 — task2 rollout + {p2_sub}")
        ax4.set_ylim(-5, 105)
        ax4.legend(loc="upper right", fontsize=7)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        n = len(bc_loss_runs)
        if n == 0:
            return
        ncols = min(5, n)
        nrows = (n + ncols - 1) // ncols
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows), squeeze=False)
        for i, losses in enumerate(bc_loss_runs):
            r, c = divmod(i, ncols)
            ax = axes2[r][c]
            it = bc_run_iters[i] if i < len(bc_run_iters) else i
            if losses:
                ax.plot(range(len(losses)), losses, "c-", lw=1.2)
            ax.set_title(f"Student BC (outer iter {it})", fontsize=8)
            ax.set_xlabel("Step", fontsize=7)
            ax.set_ylabel("Loss", fontsize=7)
            ax.grid(True, alpha=0.3)
        for j in range(n, nrows * ncols):
            r, c = divmod(j, ncols)
            axes2[r][c].set_visible(False)
        fig2.suptitle("Student BC — each training phase", fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)


def _save_results_csv_two_task(run_dir: Path, results: dict) -> None:
    csv_path = run_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k in (
            "task_id",
            "task2_id",
            "seed",
            "batch_size",
            "single_episode",
            "save_video",
            "rollout_episodes",
            "teacher_steps",
            "teacher_steps_task2",
            "teacher_lr",
            "teacher_eval_num_episodes",
            "teacher_eval_success_rate_task1",
            "teacher_eval_success_rate_task2",
            "bc_steps",
            "bc_steps_task2",
            "bc_lr",
            "bc_lr_task2",
            "max_iters",
            "max_iters_task2",
            "cumulative_buffer",
            "alignment_ratio_threshold",
            "align_min",
            "n_iters_phase1",
            "n_iters_phase2",
            "final_success_phase1",
            "final_success_phase2",
            "stopped_reason_phase1",
            "stopped_reason_phase2",
            "lora",
            "action_expert_only",
            "full_experiment",
            "phase2_eval_interval",
            "student_action_merge",
            "phase2_self_replay",
            "temporal_decay",
            "l1_bc_loss",
            "kl_lambda",
            "group_size",
            "teacher_group_size",
        ):
            if k in results:
                writer.writerow([k, results[k]])
        for ev in results.get("periodic_eval_10ep_phase1") or []:
            writer.writerow(
                [f"periodic10ep_phase1_iter{ev['iter']}_success_rate", ev["success_rate"]]
            )
        if results.get("rollout_episodes") == 1:
            writer.writerow(
                ["student_final_eval_task1_10ep", results.get("student_final_eval_task1_10ep")]
            )
            writer.writerow(
                ["student_final_eval_task2_10ep", results.get("student_final_eval_task2_10ep")]
            )
        for ev in results.get("distill_eval_phase1", []):
            writer.writerow([f"phase1_distill_iter{ev['iter']}_success_rate", ev["success_rate"]])
        for ev in results.get("phase2_by_iter", []):
            writer.writerow(
                [
                    f"phase2_iter{ev['iter']}_rollout_sr_task2",
                    ev["rollout_success_rate_task2"],
                ]
            )
            if ev.get("eval10_task1") is not None and ev.get("eval10_task2") is not None:
                writer.writerow([f"phase2_iter{ev['iter']}_eval10_task1", ev["eval10_task1"]])
                writer.writerow([f"phase2_iter{ev['iter']}_eval10_task2", ev["eval10_task2"]])


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(
        description="Two-task sequential on-policy distillation; student continues on task2"
    )
    parser.add_argument("--task", type=int, required=True, help="LIBERO task id — phase 1 (task1)")
    parser.add_argument("--task2", type=int, required=True, help="LIBERO task id — phase 2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--teacher_steps", type=int, default=50, help="Teacher FT steps on task1")
    parser.add_argument(
        "--teacher_steps_task2",
        type=int,
        default=None,
        help="Teacher FT steps on task2 (default: same as --teacher_steps)",
    )
    parser.add_argument("--teacher_lr", type=float, default=1e-4)
    parser.add_argument("--bc_steps", type=int, default=10, help="BC steps per outer iter (task1)")
    parser.add_argument(
        "--bc_steps_task2",
        type=int,
        default=None,
        help="BC steps per outer iter on task2 (default: same as --bc_steps)",
    )
    parser.add_argument("--bc_lr", type=float, default=1e-4)
    parser.add_argument(
        "--bc_lr_task2",
        type=float,
        default=None,
        help="BC LR on task2 (default: same as --bc_lr)",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=50,
        help=(
            "Outer distillation iters on task1 (stop early on rollout success unless --full_experiment)"
        ),
    )
    parser.add_argument(
        "--max_iters_task2",
        type=int,
        default=50,
        help="Outer distillation iters on task2 (stop early on rollout success unless --full_experiment)",
    )
    parser.add_argument("--teacher_eval_episodes", type=int, default=10)
    parser.add_argument("--rollout_episodes", type=int, default=1)
    parser.add_argument(
        "--save_video",
        action="store_true",
        default=True,
        help=(
            "Save videos only for phase-1 periodic 10-ep student eval and phase-2 dual 10-ep student eval "
            "(when those runs execute). Teacher FT evals, distillation rollouts, and final student eval "
            "do not write videos."
        ),
    )
    parser.add_argument(
        "--no_save_video",
        action="store_false",
        dest="save_video",
        help="Disable videos for periodic / dual 10-ep evals (see --save_video).",
    )
    parser.add_argument("--cumulative_buffer", action="store_true")
    parser.add_argument("--alignment_ratio_threshold", type=float, default=None)
    parser.add_argument("--align_min", type=float, default=None)
    parser.add_argument("--single_episode", action="store_true")
    parser.add_argument(
        "--lora",
        action="store_true",
        help=(
            "Fine-tune with LoRA (loads pi05_libero_lora checkpoint; same idea as "
            "augment_finetune_experiment.py). Mutually exclusive with --action_expert_only."
        ),
    )
    parser.add_argument(
        "--action_expert_only",
        action="store_true",
        help=(
            "Train only the action expert (freeze rest). Mutually exclusive with --lora."
        ),
    )
    parser.add_argument(
        "--full_experiment",
        action="store_true",
        help=(
            "Do not stop early when rollout success is 100%% on task1/task2; run all max_iters / "
            f"max_iters_task2. Every {FULL_EXPERIMENT_EVAL_INTERVAL} outer iterations run a separate "
            f"{FULL_EXPERIMENT_EVAL_EPISODES}-ep student eval on task1 each {FULL_EXPERIMENT_EVAL_INTERVAL} "
            "outer iters (phase 1); phase 2 runs all max_iters_task2 without early stop. "
            "See losses.pdf."
        ),
    )
    parser.add_argument(
        "--phase2_eval_interval",
        type=int,
        default=5,
        help=(
            "Phase 2: run dual 10-episode student evals (task1 and task2, no teacher) every this many "
            "outer phase-2 iterations only (no exceptions). Use 1 to run after every iteration."
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
        "--phase2_self_replay",
        action="store_true",
        help=(
            "Phase 2: augment each on-policy distillation BC batch — same replan observations as the "
            "task2 rollout + teacher targets, interleaved with paired rows that use the task1 prompt "
            "and action targets from a frozen copy of the student taken at the start of phase 2 "
            "(before teacher2 training). Single train_model_on_fly call per outer iter."
        ),
    )
    parser.add_argument(
        "--temporal_decay",
        type=float,
        default=1.0,
        help=(
            "Per-replan BC sample weight = temporal_decay ** (env step at replan) on distillation rollouts; "
            "1.0 = uniform."
        ),
    )
    parser.add_argument(
        "--l1_bc_loss",
        action="store_true",
        help="Student BC only: L1 on diffusion residual instead of L2 MSE.",
    )
    parser.add_argument(
        "--kl_lambda",
        type=float,
        default=0.0,
        help=(
            "Student BC (Pi0): add kl_lambda * MSE(v_t, stop_grad(v_t^ref)) vs frozen student snapshot "
            "at start of each BC segment; same noise/timestep as teacher diffusion loss."
        ),
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=1,
        help=(
            "Distillation rollouts: G independent student action-chunk samples per replan; first drives the env."
        ),
    )
    parser.add_argument(
        "--teacher_group_size",
        type=int,
        default=1,
        help=(
            "Distillation rollouts: T teacher noise samples per replan (T==1 uses student's first noise). "
            "Metrics in distillation_rollout_metrics.pdf per iter."
        ),
    )
    args = parser.parse_args()
    if args.alignment_ratio_threshold is not None and args.align_min is not None:
        parser.error("use only one of --alignment_ratio_threshold and --align_min")
    if args.lora and args.action_expert_only:
        parser.error("--lora and --action_expert_only are mutually exclusive")
    sam = float(args.student_action_merge)
    if not (0.0 <= sam <= 1.0):
        parser.error("--student_action_merge must be in [0, 1]")
    temporal_decay = float(args.temporal_decay)
    if temporal_decay < 0:
        parser.error("--temporal_decay must be >= 0")
    kl_lambda = float(args.kl_lambda)
    if kl_lambda < 0.0:
        parser.error("--kl_lambda must be >= 0")
    group_size = max(1, int(args.group_size))
    teacher_group_size = max(1, int(args.teacher_group_size))

    task1_id = args.task
    task2_id = args.task2
    seed = args.seed
    batch_size = args.batch_size
    teacher_steps = args.teacher_steps
    teacher_steps_t2 = args.teacher_steps_task2 or teacher_steps
    teacher_lr = args.teacher_lr
    bc_steps = args.bc_steps
    bc_steps_t2 = args.bc_steps_task2 or bc_steps
    bc_lr = args.bc_lr
    bc_lr_t2 = args.bc_lr_task2 if args.bc_lr_task2 is not None else bc_lr
    max_iters = args.max_iters
    max_iters_task2 = args.max_iters_task2
    align_th = args.alignment_ratio_threshold
    align_min = args.align_min
    rollout_episodes = max(1, int(args.rollout_episodes))
    phase2_eval_interval = max(1, int(args.phase2_eval_interval))
    repo_id = "antoniomari/libero_90"

    run_dir, video_dir, losses_pdf = _ensure_run_dir_two_task(
        task1=task1_id,
        task2=task2_id,
        seed=seed,
        teacher_lr=teacher_lr,
        teacher_steps=teacher_steps,
        teacher_steps_task2=teacher_steps_t2,
        bc_lr=bc_lr,
        bc_steps=bc_steps,
        max_iters=max_iters,
        max_iters_task2=max_iters_task2,
        single_episode=args.single_episode,
        lora=args.lora,
        action_expert_only=args.action_expert_only,
        alignment_ratio_threshold=align_th,
        align_min=align_min,
        rollout_episodes=rollout_episodes,
        full_experiment=args.full_experiment,
        phase2_eval_interval=phase2_eval_interval,
        student_action_merge=sam,
        phase2_self_replay=args.phase2_self_replay,
        temporal_decay=temporal_decay,
        l1_bc_loss=bool(args.l1_bc_loss),
        kl_lambda=kl_lambda,
        group_size=group_size,
        teacher_group_size=teacher_group_size,
    )
    print(f"Run directory: {run_dir}")
    print(f"LoRA: {args.lora}  |  Action expert only: {args.action_expert_only}")
    print(
        "Hyperparameters (effective values after defaults):\n"
        f"  task1={task1_id}  task2={task2_id}  seed={seed}\n"
        f"  batch_size={batch_size}  rollout_episodes={rollout_episodes}\n"
        f"  teacher_lr={teacher_lr}  teacher_steps={teacher_steps}  "
        f"teacher_steps_task2={teacher_steps_t2}  (CLI teacher_steps_task2={args.teacher_steps_task2!r})\n"
        f"  bc_lr={bc_lr}  bc_steps={bc_steps}  "
        f"bc_lr_task2={bc_lr_t2}  bc_steps_task2={bc_steps_t2}  "
        f"(CLI bc_lr_task2={args.bc_lr_task2!r}  bc_steps_task2={args.bc_steps_task2!r})\n"
        f"  max_iters={max_iters}  max_iters_task2={max_iters_task2}\n"
        f"  teacher_eval_episodes={args.teacher_eval_episodes}\n"
        f"  single_episode={args.single_episode}  cumulative_buffer={args.cumulative_buffer}  "
        f"save_video={args.save_video}\n"
        f"  alignment_ratio_threshold={align_th!r}  align_min={align_min!r}\n"
        f"  full_experiment={args.full_experiment}\n"
        f"  phase2_eval_interval={phase2_eval_interval}  (CLI {args.phase2_eval_interval!r})\n"
        f"  student_action_merge={sam}  (CLI {args.student_action_merge!r})\n"
        f"  phase2_self_replay={args.phase2_self_replay}\n"
        f"  temporal_decay={temporal_decay}  l1_bc_loss={args.l1_bc_loss}  kl_lambda={kl_lambda}\n"
        f"  group_size={group_size}  teacher_group_size={teacher_group_size}\n"
        f"  repo_id={repo_id}  task_suite={TASK_SUITE_NAME}\n"
        f"  argv: {sys.argv}"
    )

    print("\nLoading base model...")
    base_model, config = load_pi05_libero_model(
        use_base_model=False,
        use_lora=args.lora,
        action_expert_only=args.action_expert_only,
    )
    # Match on_policy_distillation.py (default TrainConfig num_workers, typically 2 for pi05_libero).
    config = dataclasses.replace(config, batch_size=batch_size)

    # ---------- Phase 1: teacher1 + student from base ----------
    print("\n" + "=" * 50)
    print(f"Phase 1 — Teacher: fine-tune on task {task1_id}")
    print("=" * 50)
    with override_create_torch_dataset(
        repo_id=repo_id,
        task_id=task1_id,
        mirror_data=True,
        single_episode=args.single_episode,
        augment=False,
    ):
        teacher_train_loader = _data_loader.create_data_loader(
            config, sharding=None, shuffle=True,
        )

    teacher1 = copy_model(base_model, config)
    t1_losses: list[float] = []
    t2_losses: list[float] = []

    def _on_t1(_s: int, lv: float) -> None:
        t1_losses.append(lv)

    teacher1, _, _ = train_model_on_fly(
        model=teacher1,
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
        on_step_callback=_on_t1,
    )
    teacher1.eval()
    del teacher_train_loader

    teacher1_policy = create_policy(teacher1, config, CHECKPOINT_DIR, rng_seed=seed)

    teval_n = max(1, int(args.teacher_eval_episodes))
    tev1 = video_dir / "teacher_task1_post_train_eval"
    tev1.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 50)
    print(
        f"Teacher evaluation: {teval_n} episodes on task {task1_id} "
        "(after fine-tuning)"
    )
    print("=" * 50)
    teacher_eval_sr1, _ = run_evaluation(
        policy=teacher1_policy,
        train_config=config,
        num_trials=teval_n,
        task_suite_name=TASK_SUITE_NAME,
        task_id=task1_id,
        save_video=False,
        video_out_path=str(tev1),
        seed=seed,
        show_progress_bar=True,
    )
    teacher_eval_sr2: float | None = None
    print(
        f"Teacher eval success rate: {teacher_eval_sr1 * 100:.1f}% "
        f"({int(round(teacher_eval_sr1 * teval_n))}/{teval_n} episodes)"
    )

    opd = _get_opd()

    print("\n" + "=" * 50)
    print("Phase 1 — Student: copy from base (not teacher1)")
    print("=" * 50)
    student_model = copy_model(base_model, config)
    student_model.eval()
    del base_model

    bc_buffer: list[dict] = []
    all_bc_losses: list[float] = []
    bc_offsets: list[int] = [0]
    bc_loss_runs: list[list[float]] = []
    bc_run_iters: list[int | str] = []
    distill_phase1: list[dict] = []
    phase2_by_iter: list[dict] = []
    periodic_eval_10ep_phase1: list[dict] = []
    final_ok_p1 = False
    stop_p1 = "max_iters"

    if args.full_experiment:
        p2_eval_msg = (
            f"dual {DUAL_EVAL_EPISODES}-ep evals every {phase2_eval_interval} outer iters"
            if phase2_eval_interval > 1
            else f"dual {DUAL_EVAL_EPISODES}-ep evals every outer iter"
        )
        print(
            f"Full experiment mode: no early stop on rollout success; "
            f"every {FULL_EXPERIMENT_EVAL_INTERVAL} outer iterations (phase 1) run "
            f"{FULL_EXPERIMENT_EVAL_EPISODES}-ep eval on task1 (see losses.pdf). "
            f"Phase 2 runs all {max_iters_task2} outer iters without early stop; {p2_eval_msg}."
        )

    def _flush_losses_pdf(where: str) -> None:
        _plot_two_task_pdf(
            teacher1_losses=t1_losses,
            teacher2_losses=t2_losses,
            teacher_eval_sr1=float(teacher_eval_sr1),
            teacher_eval_sr2=teacher_eval_sr2,
            bc_losses=all_bc_losses,
            bc_step_offsets=bc_offsets,
            bc_loss_runs=bc_loss_runs,
            bc_run_iters=bc_run_iters,
            phase1_distill=distill_phase1,
            phase2_metrics=phase2_by_iter,
            periodic_eval_10ep_phase1=periodic_eval_10ep_phase1 or None,
            phase2_eval_interval=phase2_eval_interval,
            pdf_path=losses_pdf,
        )
        print(f"  Updated {losses_pdf} ({where})")

    v1 = video_dir / "phase1_task1"
    v1.mkdir(parents=True, exist_ok=True)
    _flush_losses_pdf("after teacher1 FT+eval, before phase 1 distillation")

    for it in range(max_iters):
        iter_video = v1 / f"iter_{it:03d}"
        iter_video.mkdir(parents=True, exist_ok=True)
        episode_examples: list[dict] = []
        student_policy = create_policy(
            student_model, config, CHECKPOINT_DIR, rng_seed=seed + it * 100_003
        )
        success_rate, episode_rollout_metrics = run_evaluation(
            policy=student_policy,
            train_config=config,
            num_trials=rollout_episodes,
            task_suite_name=TASK_SUITE_NAME,
            task_id=task1_id,
            save_video=False,
            video_out_path=str(iter_video),
            seed=seed + it,
            teacher_policy=teacher1_policy,
            show_progress_bar=True,
            distillation_examples_out=episode_examples,
            student_action_merge=sam,
            temporal_decay=temporal_decay,
            group_size=group_size,
            teacher_group_size=teacher_group_size,
            write_auxiliary_rollout_pdfs=False,
        )
        if episode_rollout_metrics:
            _get_opd()._plot_distillation_iter_rollout_metrics_pdf(
                iter_idx=it,
                task_id=task1_id,
                episode_metrics=episode_rollout_metrics,
                pdf_path=iter_video / "distillation_rollout_metrics.pdf",
            )
        # Rollout policy is not used during BC; keeping it across train_model_on_fly stacks JIT +
        # policy state with the BC step arena and can OOM on the next iter's create_policy/run_eval.
        del student_policy
        gc.collect()

        ep_ok = success_rate >= 1.0 - 1e-6
        n_new = len(episode_examples)
        filtered_ep, n_kept, n_dropped = opd._filter_examples_by_alignment(
            episode_examples,
            alignment_ratio_threshold=align_th,
            align_min=align_min,
        )
        stripped = [opd._strip_bc_metadata(e) for e in filtered_ep]
        if args.cumulative_buffer:
            bc_buffer.extend(stripped)
        else:
            bc_buffer = list(stripped)
        distill_phase1.append({"iter": it, "success_rate": float(success_rate)})
        if ep_ok:
            final_ok_p1 = True
            stop_p1 = "rollout_success"
            _flush_losses_pdf(f"phase 1 iter {it} (rollout success)")
            if not args.full_experiment:
                break
        if not bc_buffer:
            if args.full_experiment and (it + 1) % FULL_EXPERIMENT_EVAL_INTERVAL == 0:
                pe_sr = _periodic_student_eval_10ep(
                    student_model=student_model,
                    config=config,
                    task_id=task1_id,
                    seed=seed,
                    outer_iter=it,
                    video_dir=video_dir,
                    save_video=args.save_video,
                )
                periodic_eval_10ep_phase1.append({"iter": it, "success_rate": pe_sr})
                ne = FULL_EXPERIMENT_EVAL_EPISODES
                print(
                    f"  Periodic {ne}-ep eval task1 (full_experiment): {pe_sr * 100:.1f}% "
                    f"({int(round(pe_sr * ne))}/{ne})"
                )
            _flush_losses_pdf(f"phase 1 iter {it} (no BC)")
            continue
        data_cfg = config.data.create(config.assets_dirs, config.model)
        bc_bs = min(batch_size, len(bc_buffer))
        bc_loader = NeighborsDataLoader(bc_buffer, bc_bs, data_cfg)
        bc_chunk: list[float] = []

        def _on_bc(_s: int, lv: float) -> None:
            bc_chunk.append(lv)

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
            on_step_callback=_on_bc,
            l1_loss=bool(args.l1_bc_loss),
            kl_lambda=kl_lambda,
            ref_model_for_kl=ref_for_kl,
        )
        del ref_for_kl
        student_model.eval()
        del bc_loader
        all_bc_losses.extend(bc_chunk)
        bc_offsets.append(len(all_bc_losses))
        bc_loss_runs.append(list(bc_chunk))
        bc_run_iters.append(f"t1:{it}")
        _gc_after_bc_train()
        if args.full_experiment and (it + 1) % FULL_EXPERIMENT_EVAL_INTERVAL == 0:
            pe_sr = _periodic_student_eval_10ep(
                student_model=student_model,
                config=config,
                task_id=task1_id,
                seed=seed,
                outer_iter=it,
                video_dir=video_dir,
                save_video=args.save_video,
            )
            periodic_eval_10ep_phase1.append({"iter": it, "success_rate": pe_sr})
            ne = FULL_EXPERIMENT_EVAL_EPISODES
            print(
                f"  Periodic {ne}-ep eval task1 (full_experiment): {pe_sr * 100:.1f}% "
                f"({int(round(pe_sr * ne))}/{ne})"
            )
        _flush_losses_pdf(f"phase 1 iter {it} (after BC)")

    phase2_ref_model = None
    phase2_ref_policy = None
    if args.phase2_self_replay:
        print(
            "Snapshotting student at start of phase 2 (before teacher2): reference for task1 "
            "pseudo-actions paired with phase-2 OPD batches."
        )
        phase2_ref_model = copy_model(student_model, config)
        phase2_ref_policy = create_policy(
            phase2_ref_model, config, CHECKPOINT_DIR, rng_seed=seed + 88_001
        )
    del teacher1, teacher1_policy
    gc.collect()

    # ---------- Phase 2: teacher2 from reloaded base; student unchanged ----------
    print("\n" + "=" * 50)
    print(f"Phase 2 — Teacher: fine-tune on task {task2_id} (fresh from base)")
    print("=" * 50)
    print("Reloading base checkpoint for phase 2 teacher...")
    base_model, _ = load_pi05_libero_model(
        use_base_model=False,
        use_lora=args.lora,
        action_expert_only=args.action_expert_only,
    )
    with override_create_torch_dataset(
        repo_id=repo_id,
        task_id=task2_id,
        mirror_data=True,
        single_episode=args.single_episode,
        augment=False,
    ):
        loader2 = _data_loader.create_data_loader(config, sharding=None, shuffle=True)

    teacher2 = copy_model(base_model, config)
    del base_model
    gc.collect()
    def _on_t2(_s: int, lv: float) -> None:
        t2_losses.append(lv)

    teacher2, _, _ = train_model_on_fly(
        model=teacher2,
        training_data_loader=loader2,
        config=config,
        learning_rate=teacher_lr,
        num_steps=teacher_steps_t2,
        warmup_steps=0,
        weight_decay=0.0,
        log_interval=max(1, teacher_steps_t2 // 10),
        seed=seed + 19_001,
        show_progress_bar=True,
        donate_buffers=False,
        on_step_callback=_on_t2,
    )
    teacher2.eval()
    del loader2

    teacher2_policy = create_policy(teacher2, config, CHECKPOINT_DIR, rng_seed=seed + 19_002)

    if args.phase2_self_replay and phase2_ref_policy is not None:
        print(
            "\nPhase 2 self-replay: OPD batches will interleave task2+teacher targets with the same "
            "observations under the task1 prompt and reference-student action targets."
        )

    tev2 = video_dir / "teacher_task2_post_train_eval"
    tev2.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 50)
    print(
        f"Teacher evaluation (task2): {teval_n} episodes on task {task2_id} "
        "(after fine-tuning)"
    )
    print("=" * 50)
    teacher_eval_sr2, _ = run_evaluation(
        policy=teacher2_policy,
        train_config=config,
        num_trials=teval_n,
        task_suite_name=TASK_SUITE_NAME,
        task_id=task2_id,
        save_video=False,
        video_out_path=str(tev2),
        seed=seed + 19_003,
        show_progress_bar=True,
    )
    print(
        f"Teacher eval success rate: {teacher_eval_sr2 * 100:.1f}% "
        f"({int(round(teacher_eval_sr2 * teval_n))}/{teval_n} episodes)"
    )
    del teacher2

    bc_buffer = []
    final_ok_p2 = False
    stop_p2 = "max_iters"
    v2 = video_dir / "phase2_task2"
    v2.mkdir(parents=True, exist_ok=True)
    _flush_losses_pdf("after teacher2 FT+eval, before phase 2 distillation")

    for it in range(max_iters_task2):
        iter_video = v2 / f"iter_{it:03d}"
        iter_video.mkdir(parents=True, exist_ok=True)
        episode_examples: list[dict] = []
        student_policy = create_policy(
            student_model,
            config,
            CHECKPOINT_DIR,
            rng_seed=seed + 200_000 + it * 100_003,
        )
        success_rate_t2, episode_rollout_metrics_t2 = run_evaluation(
            policy=student_policy,
            train_config=config,
            num_trials=rollout_episodes,
            task_suite_name=TASK_SUITE_NAME,
            task_id=task2_id,
            save_video=False,
            video_out_path=str(iter_video),
            seed=seed + 200_000 + it,
            teacher_policy=teacher2_policy,
            show_progress_bar=True,
            distillation_examples_out=episode_examples,
            student_action_merge=sam,
            temporal_decay=temporal_decay,
            group_size=group_size,
            teacher_group_size=teacher_group_size,
            write_auxiliary_rollout_pdfs=False,
        )
        if episode_rollout_metrics_t2:
            _get_opd()._plot_distillation_iter_rollout_metrics_pdf(
                iter_idx=it,
                task_id=task2_id,
                episode_metrics=episode_rollout_metrics_t2,
                pdf_path=iter_video / "distillation_rollout_metrics.pdf",
            )
        del student_policy
        gc.collect()

        ep_ok = success_rate_t2 >= 1.0 - 1e-6
        n_new = len(episode_examples)
        filtered_ep, n_kept, n_dropped = opd._filter_examples_by_alignment(
            episode_examples,
            alignment_ratio_threshold=align_th,
            align_min=align_min,
        )
        stripped = [opd._strip_bc_metadata(e) for e in filtered_ep]
        if args.cumulative_buffer:
            bc_buffer.extend(stripped)
        else:
            bc_buffer = list(stripped)

        def _append_phase2_metrics() -> None:
            do_dual = phase2_eval_interval <= 1 or (it + 1) % phase2_eval_interval == 0
            ev1: float | None
            ev2: float | None
            if do_dual:
                ev1, ev2 = _dual_eval_student_10ep(
                    student_model=student_model,
                    config=config,
                    task1_id=task1_id,
                    task2_id=task2_id,
                    seed=seed,
                    outer_iter=it,
                    video_dir=video_dir,
                    save_video=args.save_video,
                )
            else:
                ev1, ev2 = None, None
            phase2_by_iter.append(
                {
                    "iter": it,
                    "rollout_success_rate_task2": float(success_rate_t2),
                    "eval10_task1": ev1,
                    "eval10_task2": ev2,
                }
            )
            if do_dual:
                print(
                    f"  Phase2 iter {it}: rollout_SR_task2={success_rate_t2:.2f}, "
                    f"eval10 task1={ev1:.2f}, eval10 task2={ev2:.2f}"
                )
            else:
                print(
                    f"  Phase2 iter {it}: rollout_SR_task2={success_rate_t2:.2f} "
                    f"(dual {DUAL_EVAL_EPISODES}-ep eval skipped; runs every {phase2_eval_interval} iters)"
                )

        if ep_ok:
            final_ok_p2 = True
            stop_p2 = "rollout_success"
            if not args.full_experiment:
                # Early exit: record metrics once here. With --full_experiment, continue and
                # append once at end of iter (no BC / after BC) so dual eval follows interval only.
                _append_phase2_metrics()
                _flush_losses_pdf(f"phase 2 iter {it} (rollout success)")
                break

        if not bc_buffer:
            _append_phase2_metrics()
            _flush_losses_pdf(f"phase 2 iter {it} (no BC)")
            continue

        data_cfg = config.data.create(config.assets_dirs, config.model)
        bc_bs = min(batch_size, len(bc_buffer))
        bc_chunk: list[float] = []

        def _on_bc2(_s: int, lv: float) -> None:
            bc_chunk.append(lv)

        ref_for_kl = None
        if kl_lambda > 0.0:
            ref_for_kl = copy_model(student_model, config)

        if args.phase2_self_replay and phase2_ref_policy is not None:
            task1_prompt = LIBERO_90_TASK_IDS_PROMPTS.get(task1_id)
            if not task1_prompt:
                raise ValueError(
                    f"LIBERO_90_TASK_IDS_PROMPTS has no string for task1_id={task1_id}"
                )
            aug_buffer = _phase2_opd_paired_task1_pseudo_examples(
                bc_buffer,
                phase2_ref_policy,
                task1_prompt,
                model_action_horizon=int(student_model.action_horizon),
                model_action_dim=int(student_model.action_dim),
            )
            pair_bs = max(1, bc_bs // 2)
            bc_loader = PairedOnPolicySelfReplayDataLoader(
                bc_buffer, aug_buffer, pair_bs, data_cfg
            )
            bc_train_kw: dict[str, Any] = dict(
                model=student_model,
                training_data_loader=bc_loader,
                config=config,
                learning_rate=bc_lr_t2,
                num_steps=bc_steps_t2,
                warmup_steps=0,
                weight_decay=0.0,
                log_interval=max(1, bc_steps_t2 // 5),
                seed=seed + 200_000 + it + 777,
                show_progress_bar=True,
                donate_buffers=False,
                on_step_callback=_on_bc2,
                pair_stream_layout=PAIR_STREAM_INTERLEAVED,
            )
            if align_th is not None:
                bc_train_kw["alignment_ratio_threshold"] = align_th
                bc_train_kw["alignment_reference_policy"] = phase2_ref_policy
                bc_train_kw["alignment_weight_task1_only"] = True
            bc_train_kw["l1_loss"] = bool(args.l1_bc_loss)
            bc_train_kw["kl_lambda"] = kl_lambda
            bc_train_kw["ref_model_for_kl"] = ref_for_kl
            student_model, _, _ = train_model_on_fly(**bc_train_kw)
        else:
            bc_loader = NeighborsDataLoader(bc_buffer, bc_bs, data_cfg)
            student_model, _, _ = train_model_on_fly(
                model=student_model,
                training_data_loader=bc_loader,
                config=config,
                learning_rate=bc_lr_t2,
                num_steps=bc_steps_t2,
                warmup_steps=0,
                weight_decay=0.0,
                log_interval=max(1, bc_steps_t2 // 5),
                seed=seed + 200_000 + it + 777,
                show_progress_bar=True,
                donate_buffers=False,
                on_step_callback=_on_bc2,
                l1_loss=bool(args.l1_bc_loss),
                kl_lambda=kl_lambda,
                ref_model_for_kl=ref_for_kl,
            )
        del ref_for_kl
        student_model.eval()
        del bc_loader
        all_bc_losses.extend(bc_chunk)
        bc_offsets.append(len(all_bc_losses))
        bc_loss_runs.append(list(bc_chunk))
        bc_run_iters.append(f"t2:{it}")

        _gc_after_bc_train()
        _append_phase2_metrics()
        _flush_losses_pdf(f"phase 2 iter {it} (after BC)")

    del teacher2_policy
    if phase2_ref_model is not None:
        del phase2_ref_model, phase2_ref_policy

    fin_t1: float | None = None
    fin_t2: float | None = None
    if rollout_episodes == 1:
        print("\nFinal student eval (10 ep) on task1 and task2")
        fe = video_dir / f"student_final_eval_{STUDENT_FINAL_EVAL_EPISODES}ep"
        p1 = create_policy(student_model, config, CHECKPOINT_DIR, rng_seed=seed + 91_011_013)
        fin_t1, _ = run_evaluation(
            policy=p1,
            train_config=config,
            num_trials=STUDENT_FINAL_EVAL_EPISODES,
            task_suite_name=TASK_SUITE_NAME,
            task_id=task1_id,
            save_video=False,
            video_out_path=str(fe / "task1"),
            seed=seed + 91_011_014,
            teacher_policy=None,
            show_progress_bar=True,
        )
        p2 = create_policy(student_model, config, CHECKPOINT_DIR, rng_seed=seed + 91_011_015)
        fin_t2, _ = run_evaluation(
            policy=p2,
            train_config=config,
            num_trials=STUDENT_FINAL_EVAL_EPISODES,
            task_suite_name=TASK_SUITE_NAME,
            task_id=task2_id,
            save_video=False,
            video_out_path=str(fe / "task2"),
            seed=seed + 91_011_016,
            teacher_policy=None,
            show_progress_bar=True,
        )
        fin_t1, fin_t2 = float(fin_t1), float(fin_t2)
        print(f"  Final task1: {fin_t1 * 100:.1f}%, task2: {fin_t2 * 100:.1f}%")

    n_p1 = len(distill_phase1)
    results = {
        "task_id": task1_id,
        "task2_id": task2_id,
        "seed": seed,
        "batch_size": batch_size,
        "single_episode": args.single_episode,
        "save_video": args.save_video,
        "rollout_episodes": rollout_episodes,
        "teacher_steps": teacher_steps,
        "teacher_steps_task2": teacher_steps_t2,
        "teacher_lr": teacher_lr,
        "teacher_eval_num_episodes": teval_n,
        "teacher_eval_success_rate_task1": float(teacher_eval_sr1),
        "teacher_eval_success_rate_task2": float(teacher_eval_sr2),
        "bc_steps": bc_steps,
        "bc_steps_task2": bc_steps_t2,
        "bc_lr": bc_lr,
        "bc_lr_task2": bc_lr_t2,
        "max_iters": max_iters,
        "max_iters_task2": max_iters_task2,
        "cumulative_buffer": args.cumulative_buffer,
        "alignment_ratio_threshold": align_th,
        "align_min": align_min,
        "n_iters_phase1": n_p1,
        "n_iters_phase2": len(phase2_by_iter),
        "final_success_phase1": final_ok_p1,
        "final_success_phase2": final_ok_p2,
        "stopped_reason_phase1": stop_p1,
        "stopped_reason_phase2": stop_p2,
        "lora": args.lora,
        "action_expert_only": args.action_expert_only,
        "full_experiment": args.full_experiment,
        "phase2_eval_interval": phase2_eval_interval,
        "student_action_merge": sam,
        "phase2_self_replay": args.phase2_self_replay,
        "temporal_decay": temporal_decay,
        "l1_bc_loss": bool(args.l1_bc_loss),
        "kl_lambda": kl_lambda,
        "group_size": group_size,
        "teacher_group_size": teacher_group_size,
        "periodic_eval_10ep_phase1": periodic_eval_10ep_phase1,
        "distill_eval_phase1": distill_phase1,
        "phase2_by_iter": phase2_by_iter,
        "student_final_eval_task1_10ep": fin_t1,
        "student_final_eval_task2_10ep": fin_t2,
    }
    _save_results_csv_two_task(run_dir, results)

    _flush_losses_pdf("final")


if __name__ == "__main__":
    main()
