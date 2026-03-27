## Sequential two-task on-policy distillation.
#
# Phase 1 (task): teacher from base -> FT on task1; student from base -> distillation on task1.
# Phase 2 (task2): new teacher from base -> FT on task2; same student weights (no reset) ->
# distillation on task2. Each outer iteration on task2: rollout on task2 for BC, then separate
# 10-episode evals on task1 and task2 (no teacher).
#
# LoRA / action-expert-only: same as augment_finetune_experiment.py — pass --lora to load
# pi05_libero_lora and fine-tune adapters; or --action_expert_only (mutually exclusive with --lora).
# Run dirs: on_policy_distillation_two_task[_single][_lora|_action_expert]/...
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

# Reuse helpers from single-task script (alignment plots, filtering, logging).
_OP_PATH = Path(__file__).resolve().parent / "on_policy_distillation.py"
_spec = importlib.util.spec_from_file_location("_opd_single", _OP_PATH)
assert _spec and _spec.loader
_opd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_opd)

DUAL_EVAL_EPISODES = 10
STUDENT_FINAL_EVAL_EPISODES = _opd.STUDENT_FINAL_EVAL_EPISODES
TASK_SUITE_NAME = _opd.TASK_SUITE_NAME
CHECKPOINT_DIR = _opd.CHECKPOINT_DIR


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
) -> tuple[Path, Path, Path]:
    folder_base = "on_policy_distillation_two_task_single" if single_episode else "on_policy_distillation_two_task"
    if lora:
        folder_base = f"{folder_base}_lora"
    elif action_expert_only:
        folder_base = f"{folder_base}_action_expert"
    parent = _results_root() / folder_base / f"task{task1}_task{task2}_seed{seed}"
    base_name = (
        f"tlr{teacher_lr}_ts{teacher_steps}_ts2{teacher_steps_task2}_bc_lr{bc_lr}_steps{bc_steps}_"
        f"mi{max_iters}_mi2{max_iters_task2}"
    )
    if rollout_episodes != 1:
        base_name = f"{base_name}_rolloutEp{rollout_episodes}"
    if alignment_ratio_threshold is not None:
        base_name = f"{base_name}_align{alignment_ratio_threshold}"
    elif align_min is not None:
        base_name = f"{base_name}_alignmin{align_min}"
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


def _plot_two_task_pdf(
    *,
    teacher1_losses: list[float],
    teacher2_losses: list[float],
    bc_losses: list[float],
    bc_step_offsets: list[int],
    bc_loss_runs: list[list[float]],
    bc_run_iters: list[int | str],
    phase1_distill: list[dict],
    phase2_metrics: list[dict],
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
        ax0.set_title("Teacher fine-tune — task 1")
        ax0.legend(loc="upper right", fontsize=8)
        ax0.grid(True, alpha=0.3)

        if teacher2_losses:
            ax1.plot(range(len(teacher2_losses)), teacher2_losses, "navy", label="Teacher task2", lw=1.5)
        ax1.set_ylabel("Loss")
        ax1.set_xlabel("Gradient step")
        ax1.set_title("Teacher fine-tune — task 2 (from base)")
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

        if phase1_distill:
            iters = [int(x["iter"]) for x in phase1_distill]
            srs = [float(x["success_rate"]) * 100.0 for x in phase1_distill]
            ax3.plot(iters, srs, "g-o", label="Rollout SR (task1 phase)", lw=1.2, ms=4)
        ax3.set_ylabel("Success %")
        ax3.set_xlabel("Outer iter (task 1)")
        ax3.set_title("Phase 1 — distillation rollout success (task 1)")
        ax3.set_ylim(-5, 105)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)

        if phase2_metrics:
            it2 = [int(x["iter"]) for x in phase2_metrics]
            r2 = [float(x["rollout_success_rate_task2"]) * 100.0 for x in phase2_metrics]
            e1 = [float(x["eval10_task1"]) * 100.0 for x in phase2_metrics]
            e2 = [float(x["eval10_task2"]) * 100.0 for x in phase2_metrics]
            ax4.plot(it2, r2, "g-s", label="Rollout SR (task2, training)", lw=1.2, ms=4)
            ax4.plot(it2, e1, "C1-^", label=f"Eval {DUAL_EVAL_EPISODES}ep task1", lw=1.2, ms=5)
            ax4.plot(it2, e2, "C2-d", label=f"Eval {DUAL_EVAL_EPISODES}ep task2", lw=1.2, ms=5)
        ax4.set_ylabel("Success %")
        ax4.set_xlabel("Outer iter (task 2)")
        ax4.set_title(
            f"Phase 2 — task2 rollout + separate {DUAL_EVAL_EPISODES}-ep evals on both tasks"
        )
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
        ):
            if k in results:
                writer.writerow([k, results[k]])
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
            writer.writerow([f"phase2_iter{ev['iter']}_eval10_task1", ev["eval10_task1"]])
            writer.writerow([f"phase2_iter{ev['iter']}_eval10_task2", ev["eval10_task2"]])


def main() -> None:
    _opd._setup_logging()
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
    parser.add_argument("--max_iters", type=int, default=50, help="Outer distillation iters on task1")
    parser.add_argument(
        "--max_iters_task2",
        type=int,
        default=50,
        help="Outer distillation iters on task2",
    )
    parser.add_argument("--teacher_eval_episodes", type=int, default=10)
    parser.add_argument("--rollout_episodes", type=int, default=1)
    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument("--no_save_video", action="store_false", dest="save_video")
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
    args = parser.parse_args()
    if args.alignment_ratio_threshold is not None and args.align_min is not None:
        parser.error("use only one of --alignment_ratio_threshold and --align_min")
    if args.lora and args.action_expert_only:
        parser.error("--lora and --action_expert_only are mutually exclusive")

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
    )
    print(f"Run directory: {run_dir}")
    print(f"LoRA: {args.lora}  |  Action expert only: {args.action_expert_only}")

    print("\nLoading base model...")
    base_model, config = load_pi05_libero_model(
        use_base_model=False,
        use_lora=args.lora,
        action_expert_only=args.action_expert_only,
    )
    # num_workers=0: JAX is used in dataset transforms (e.g. mirror); worker processes
    # often cannot initialize CUDA and crash with "no supported devices for platform CUDA".
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
        loader1 = _data_loader.create_data_loader(config, sharding=None, shuffle=True)

    teacher1 = copy_model(base_model, config)
    t1_losses: list[float] = []

    def _on_t1(_s: int, lv: float) -> None:
        t1_losses.append(lv)

    teacher1, _, _ = train_model_on_fly(
        model=teacher1,
        training_data_loader=loader1,
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
    del loader1
    teacher1_policy = create_policy(teacher1, config, CHECKPOINT_DIR, rng_seed=seed)

    teval_n = max(1, int(args.teacher_eval_episodes))
    tev1 = video_dir / "teacher_task1_post_train_eval"
    tev1.mkdir(parents=True, exist_ok=True)
    teacher_eval_sr1, _ = run_evaluation(
        policy=teacher1_policy,
        train_config=config,
        num_trials=teval_n,
        task_suite_name=TASK_SUITE_NAME,
        task_id=task1_id,
        save_video=args.save_video,
        video_out_path=str(tev1),
        seed=seed,
        show_progress_bar=True,
    )
    print(f"Teacher1 eval on task {task1_id}: {teacher_eval_sr1 * 100:.1f}%")

    print("\n" + "=" * 50)
    print("Phase 1 — Student: copy from base (not teacher1)")
    print("=" * 50)
    student_model = copy_model(base_model, config)
    student_model.eval()

    bc_buffer: list[dict] = []
    all_bc_losses: list[float] = []
    bc_offsets: list[int] = [0]
    bc_loss_runs: list[list[float]] = []
    bc_run_iters: list[int | str] = []
    distill_phase1: list[dict] = []
    final_ok_p1 = False
    stop_p1 = "max_iters"

    v1 = video_dir / "phase1_task1"
    v1.mkdir(parents=True, exist_ok=True)

    for it in range(max_iters):
        iter_video = v1 / f"iter_{it:03d}"
        iter_video.mkdir(parents=True, exist_ok=True)
        episode_examples: list[dict] = []
        student_policy = create_policy(
            student_model, config, CHECKPOINT_DIR, rng_seed=seed + it * 100_003
        )
        success_rate, _ = run_evaluation(
            policy=student_policy,
            train_config=config,
            num_trials=rollout_episodes,
            task_suite_name=TASK_SUITE_NAME,
            task_id=task1_id,
            save_video=args.save_video,
            video_out_path=str(iter_video),
            seed=seed + it,
            teacher_policy=teacher1_policy,
            show_progress_bar=True,
            distillation_examples_out=episode_examples,
        )
        ep_ok = success_rate >= 1.0 - 1e-6
        n_new = len(episode_examples)
        ratios = [_opd._example_alignment_ratio(e) for e in episode_examples]
        env_steps: list[int] = [
            j if (s := _opd._example_env_step(e)) is None else int(s)
            for j, e in enumerate(episode_examples)
        ]
        filtered_ep, n_kept, n_dropped = _opd._filter_examples_by_alignment(
            episode_examples,
            alignment_ratio_threshold=align_th,
            align_min=align_min,
        )
        stripped = [_opd._strip_bc_metadata(e) for e in filtered_ep]
        if ratios:
            _opd._plot_alignment_ratios_iter_pdf(
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
            bc_buffer.extend(stripped)
        else:
            bc_buffer = list(stripped)
        distill_phase1.append({"iter": it, "success_rate": float(success_rate)})
        if ep_ok:
            final_ok_p1 = True
            stop_p1 = "rollout_success"
            break
        if not bc_buffer:
            continue
        data_cfg = config.data.create(config.assets_dirs, config.model)
        bc_bs = min(batch_size, len(bc_buffer))
        bc_loader = NeighborsDataLoader(bc_buffer, bc_bs, data_cfg)
        bc_chunk: list[float] = []

        def _on_bc(_s: int, lv: float) -> None:
            bc_chunk.append(lv)

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
        )
        student_model.eval()
        all_bc_losses.extend(bc_chunk)
        bc_offsets.append(len(all_bc_losses))
        bc_loss_runs.append(list(bc_chunk))
        bc_run_iters.append(f"t1:{it}")
        gc.collect()

    del teacher1, teacher1_policy

    # ---------- Phase 2: teacher2 from base; student unchanged ----------
    print("\n" + "=" * 50)
    print(f"Phase 2 — Teacher: fine-tune on task {task2_id} (fresh from base)")
    print("=" * 50)
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
    t2_losses: list[float] = []

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

    tev2 = video_dir / "teacher_task2_post_train_eval"
    tev2.mkdir(parents=True, exist_ok=True)
    teacher_eval_sr2, _ = run_evaluation(
        policy=teacher2_policy,
        train_config=config,
        num_trials=teval_n,
        task_suite_name=TASK_SUITE_NAME,
        task_id=task2_id,
        save_video=args.save_video,
        video_out_path=str(tev2),
        seed=seed + 19_003,
        show_progress_bar=True,
    )
    print(f"Teacher2 eval on task {task2_id}: {teacher_eval_sr2 * 100:.1f}%")
    del teacher2

    bc_buffer = []
    phase2_by_iter: list[dict] = []
    final_ok_p2 = False
    stop_p2 = "max_iters"
    v2 = video_dir / "phase2_task2"
    v2.mkdir(parents=True, exist_ok=True)

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
        success_rate_t2, _ = run_evaluation(
            policy=student_policy,
            train_config=config,
            num_trials=rollout_episodes,
            task_suite_name=TASK_SUITE_NAME,
            task_id=task2_id,
            save_video=args.save_video,
            video_out_path=str(iter_video),
            seed=seed + 200_000 + it,
            teacher_policy=teacher2_policy,
            show_progress_bar=True,
            distillation_examples_out=episode_examples,
        )
        ep_ok = success_rate_t2 >= 1.0 - 1e-6
        n_new = len(episode_examples)
        ratios = [_opd._example_alignment_ratio(e) for e in episode_examples]
        env_steps: list[int] = [
            j if (s := _opd._example_env_step(e)) is None else int(s)
            for j, e in enumerate(episode_examples)
        ]
        filtered_ep, n_kept, n_dropped = _opd._filter_examples_by_alignment(
            episode_examples,
            alignment_ratio_threshold=align_th,
            align_min=align_min,
        )
        stripped = [_opd._strip_bc_metadata(e) for e in filtered_ep]
        if ratios:
            _opd._plot_alignment_ratios_iter_pdf(
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
            bc_buffer.extend(stripped)
        else:
            bc_buffer = list(stripped)

        def _append_phase2_metrics() -> None:
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
            phase2_by_iter.append(
                {
                    "iter": it,
                    "rollout_success_rate_task2": float(success_rate_t2),
                    "eval10_task1": ev1,
                    "eval10_task2": ev2,
                }
            )
            print(
                f"  Phase2 iter {it}: rollout_SR_task2={success_rate_t2:.2f}, "
                f"eval10 task1={ev1:.2f}, eval10 task2={ev2:.2f}"
            )

        if ep_ok:
            _append_phase2_metrics()
            final_ok_p2 = True
            stop_p2 = "rollout_success"
            break

        if not bc_buffer:
            _append_phase2_metrics()
            continue

        data_cfg = config.data.create(config.assets_dirs, config.model)
        bc_bs = min(batch_size, len(bc_buffer))
        bc_loader = NeighborsDataLoader(bc_buffer, bc_bs, data_cfg)
        bc_chunk: list[float] = []

        def _on_bc2(_s: int, lv: float) -> None:
            bc_chunk.append(lv)

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
        )
        student_model.eval()
        all_bc_losses.extend(bc_chunk)
        bc_offsets.append(len(all_bc_losses))
        bc_loss_runs.append(list(bc_chunk))
        bc_run_iters.append(f"t2:{it}")
        gc.collect()
        _append_phase2_metrics()

    del teacher2_policy

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
            save_video=args.save_video,
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
            save_video=args.save_video,
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
        "distill_eval_phase1": distill_phase1,
        "phase2_by_iter": phase2_by_iter,
        "student_final_eval_task1_10ep": fin_t1,
        "student_final_eval_task2_10ep": fin_t2,
    }
    _save_results_csv_two_task(run_dir, results)

    _plot_two_task_pdf(
        teacher1_losses=t1_losses,
        teacher2_losses=t2_losses,
        bc_losses=all_bc_losses,
        bc_step_offsets=bc_offsets,
        bc_loss_runs=bc_loss_runs,
        bc_run_iters=bc_run_iters,
        phase1_distill=distill_phase1,
        phase2_metrics=phase2_by_iter,
        pdf_path=losses_pdf,
    )
    print(f"\nSaved {losses_pdf}")


if __name__ == "__main__":
    main()
