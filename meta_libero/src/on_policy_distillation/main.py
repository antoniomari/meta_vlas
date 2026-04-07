"""CLI entry: single-task on-policy distillation."""

from __future__ import annotations

import argparse
import dataclasses
import gc
from typing import Any

import numpy as np
import wandb
from openpi.training import data_loader as _data_loader  # type: ignore

from meta_libero.src.dataset import override_create_torch_dataset  # type: ignore
from meta_libero.src.on_policy_distillation.bc_data import (
    example_alignment_ratio,
    filter_examples_by_alignment,
    filter_examples_by_max_teacher_variance,
    strip_bc_metadata,
)
from meta_libero.src.on_policy_distillation.constants import (
    CHECKPOINT_DIR,
    FULL_EXPERIMENT_EVAL_EPISODES,
    FULL_EXPERIMENT_EVAL_INTERVAL,
    STUDENT_FINAL_EVAL_EPISODES,
    TASK_SUITE_NAME,
)
from meta_libero.src.on_policy_distillation.logging_utils import setup_logging
from meta_libero.src.on_policy_distillation.metrics import trainable_param_delta_l2_norm
from meta_libero.src.on_policy_distillation.periodic_eval import periodic_student_eval_10ep
from meta_libero.src.on_policy_distillation.plotting import (
    plot_distillation_iter_rollout_metrics_pdf,
    plot_losses_pdf,
)
from meta_libero.src.on_policy_distillation.results_csv import save_results_csv
from meta_libero.src.on_policy_distillation.run_layout import ensure_run_dir
from meta_libero.src.on_policy_distillation.wandb_run import init_wandb, summarize_rollout_metrics
from meta_libero.src.ttt import (  # type: ignore
    NeighborsDataLoader,
    copy_model,
    create_policy,
    load_pi05_libero_model,
    run_evaluation,
    train_model_on_fly,
)


def main() -> None:
    setup_logging()
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
    parser.add_argument(
        "--grpo_trust_eps",
        type=float,
        default=None,
        help=(
            "GRPO only: trust-region BC target per sample: "
            "a_target = a_stud + clamp(a_teach - a_stud, -eps, eps) (element-wise). "
            "Default None uses raw student chunk a_i as the regression target."
        ),
    )
    parser.add_argument(
        "--grpo_weight",
        type=str,
        default="none",
        choices=("none", "mean_std"),
        help=(
            "GRPO only: optional per-step group multiplier on BC sample weights. "
            "'none' = advantage × temporal decay only. "
            "'mean_std' multiplies by mean(||a_i-a_teach||)/(std(||·||)+eps) for the group "
            "(amplifies when the group is confidently wrong vs spread out)."
        ),
    )
    parser.add_argument(
        "--grpo_weight_eps",
        type=float,
        default=1e-8,
        help="Denominator epsilon for --grpo_weight mean_std (must be > 0).",
    )
    parser.add_argument(
        "--distill_collect_every",
        type=int,
        default=1,
        help=(
            "Distillation rollout only: append BC rows every N environment steps after "
            "--num_steps_wait (condition (t-num_steps_wait) %% N == 0). Default 1 = every step."
        ),
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help=(
            "If set, log metrics to Weights & Biases (project name). "
            "Authenticate with `wandb login` or set env WANDB_API_KEY (see script docstring)."
        ),
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Optional W&B entity (team username or org).",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional run name (default: run directory leaf name).",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Optional W&B group (e.g. sweep name).",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Optional comma-separated tags for the run.",
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
    if args.grpo_trust_eps is not None:
        if not args.grpo_like:
            parser.error("--grpo_trust_eps requires --grpo_like")
        if float(args.grpo_trust_eps) <= 0.0:
            parser.error("--grpo_trust_eps must be > 0 when set")
    if args.grpo_weight != "none":
        if not args.grpo_like:
            parser.error("--grpo_weight other than none requires --grpo_like")
    if float(args.grpo_weight_eps) <= 0.0:
        parser.error("--grpo_weight_eps must be > 0")
    if int(args.distill_collect_every) < 1:
        parser.error("--distill_collect_every must be >= 1")
    distill_collect_every = int(args.distill_collect_every)
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

    run_dir, video_dir, losses_pdf = ensure_run_dir(
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
        grpo_trust_eps=args.grpo_trust_eps,
        grpo_weight=(args.grpo_weight if args.grpo_weight != "none" else None),
        grpo_weight_eps=float(args.grpo_weight_eps),
        distill_collect_every=distill_collect_every,
    )

    print(f"Run directory: {run_dir}")

    wandb_cfg: dict[str, Any] = {
        "task_id": task_id,
        "seed": seed,
        "teacher_lr": teacher_lr,
        "teacher_steps": teacher_steps,
        "bc_lr": bc_lr,
        "bc_steps": bc_steps,
        "max_iters": max_iters,
        "batch_size": batch_size,
        "rollout_episodes": rollout_episodes,
        "student_action_merge": sam,
        "group_size": group_size,
        "teacher_group_size": teacher_group_size,
        "temporal_decay": temporal_decay,
        "l1_bc_loss": bool(args.l1_bc_loss),
        "kl_lambda": kl_lambda,
        "max_teacher_variance": max_teacher_var,
        "grpo_like": bool(args.grpo_like),
        "grpo_trust_eps": args.grpo_trust_eps,
        "grpo_weight": args.grpo_weight,
        "grpo_weight_eps": float(args.grpo_weight_eps),
        "distill_collect_every": distill_collect_every,
        "student_pretraining_steps": student_pretraining_steps,
        "cumulative_buffer": args.cumulative_buffer,
        "alignment_ratio_threshold": align_th,
        "align_min": align_min,
        "full_experiment": args.full_experiment,
        "single_episode": args.single_episode,
        "lora": args.lora,
        "action_expert_only": args.action_expert_only,
        "run_dir": str(run_dir),
    }
    wb_run = init_wandb(args, run_dir, config=wandb_cfg)

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
        if wb_run is not None:
            wandb.log({"teacher_step": _step, "teacher/loss": float(loss_val)})

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
    if wb_run is not None:
        wandb.log(
            {
                "teacher_step": teacher_steps,
                "teacher/eval_success_rate": float(teacher_eval_sr),
            }
        )

    student_pretrain_losses: list[float] = []
    student_pretrain_eval: list[dict] = []

    bc_buffer: list[dict] = []
    all_bc_losses: list[float] = []
    bc_offsets: list[int] = [0]
    bc_loss_runs: list[list[float]] = []
    bc_run_iters: list[int] = []
    bc_param_delta_l2_by_iter: list[float] = []
    bc_param_delta_iters: list[int] = []
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
        plot_losses_pdf(
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
            bc_param_delta_l2_by_iter=list(bc_param_delta_l2_by_iter)
            if bc_param_delta_l2_by_iter
            else None,
            bc_param_delta_iters=list(bc_param_delta_iters) if bc_param_delta_iters else None,
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
            chunk_losses = student_pretrain_losses[-chunk:] if student_pretrain_losses else []
            mean_chunk = float(np.mean(chunk_losses)) if chunk_losses else float("nan")
            if wb_run is not None:
                wandb.log(
                    {
                        "sft_step": spt_done,
                        "sft/chunk_mean_loss": mean_chunk,
                        "sft/eval_success_rate": float(sr),
                    }
                )
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

    bc_wandb_step = 0
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
            if wb_run is not None:
                wandb.log(
                    {
                        "outer_iter": it,
                        "distill/rollout_success_rate_running": float(sr_run),
                        "distill/rollout_episodes_completed": int(ep_idx + 1),
                    }
                )
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
            grpo_trust_eps=args.grpo_trust_eps,
            grpo_weight=(
                args.grpo_weight if args.grpo_weight != "none" else None
            ),
            grpo_weight_eps=float(args.grpo_weight_eps),
            distill_collect_every=distill_collect_every,
        )
        if episode_rollout_metrics:
            plot_distillation_iter_rollout_metrics_pdf(
                iter_idx=it,
                task_id=task_id,
                episode_metrics=episode_rollout_metrics,
                pdf_path=iter_video / "distillation_rollout_metrics.pdf",
            )
        # Stop only when every rollout episode succeeded (mean rate == 1.0).
        ep_success = success_rate >= 1.0 - 1e-6
        n_new = len(episode_examples)
        ratios = [example_alignment_ratio(e) for e in episode_examples]
        filtered_ep, n_kept_al, n_drop_al = filter_examples_by_alignment(
            episode_examples,
            alignment_ratio_threshold=align_th,
            align_min=align_min,
        )
        filtered_ep, n_kept_tv, n_drop_tv = filter_examples_by_max_teacher_variance(
            filtered_ep,
            max_teacher_variance=max_teacher_var,
        )
        stripped_for_bc = [strip_bc_metadata(e) for e in filtered_ep]
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
        if wb_run is not None:
            wlog: dict[str, Any] = {
                "outer_iter": it,
                "distill/rollout_success_rate": float(success_rate),
                "distill/bc_buffer_size": int(len(bc_buffer)),
                "distill/n_new_samples": int(n_new),
                "distill/n_kept_align": int(n_kept_al),
                "distill/n_drop_align": int(n_drop_al),
                "distill/n_kept_teacher_var": int(n_kept_tv),
                "distill/n_drop_teacher_var": int(n_drop_tv),
                "distill/rollout_episode_success": float(ep_success),
            }
            wlog.update(summarize_rollout_metrics(episode_rollout_metrics))
            wandb.log(wlog)

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
                pe_sr = periodic_student_eval_10ep(
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
                if wb_run is not None:
                    wandb.log(
                        {"outer_iter": it, "distill/periodic_10ep_success_rate": float(pe_sr)}
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
            nonlocal bc_wandb_step
            bc_loss_chunk.append(loss_val)
            if wb_run is not None:
                wandb.log({"bc_step": bc_wandb_step, "bc/loss": float(loss_val)})
                bc_wandb_step += 1

        ref_for_kl = None
        if kl_lambda > 0.0:
            ref_for_kl = copy_model(student_model, config)
        # Pre-BC snapshot for ||Δθ||; reuse KL ref when present (same frozen copy).
        student_bc_snapshot = ref_for_kl if ref_for_kl is not None else copy_model(student_model, config)
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
        delta_l2 = trainable_param_delta_l2_norm(student_bc_snapshot, student_model, config)
        del student_bc_snapshot
        bc_param_delta_l2_by_iter.append(delta_l2)
        bc_param_delta_iters.append(it)
        print(f"  BC trainable weight change (L2 norm ||Δθ||): {delta_l2:.6g}")
        if wb_run is not None:
            wandb.log({"outer_iter": it, "distill/bc_param_delta_l2": float(delta_l2)})
        all_bc_losses.extend(bc_loss_chunk)
        bc_offsets.append(len(all_bc_losses))
        bc_loss_runs.append(list(bc_loss_chunk))
        bc_run_iters.append(it)
        gc.collect()
        _flush_losses_pdf(f"outer iter {it}, after student BC")

        if args.full_experiment and (it + 1) % FULL_EXPERIMENT_EVAL_INTERVAL == 0:
            pe_sr = periodic_student_eval_10ep(
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
            if wb_run is not None:
                wandb.log(
                    {"outer_iter": it, "distill/periodic_10ep_success_rate": float(pe_sr)}
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
        if wb_run is not None:
            wandb.log(
                {
                    "outer_iter": len(distill_eval_by_iter),
                    "distill/student_final_10ep_success_rate": float(student_final_eval_10ep_sr),
                }
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
        "grpo_trust_eps": float(args.grpo_trust_eps) if args.grpo_trust_eps is not None else None,
        "grpo_weight": args.grpo_weight,
        "grpo_weight_eps": float(args.grpo_weight_eps),
        "distill_collect_every": distill_collect_every,
        "student_pretraining_steps": student_pretraining_steps,
        "student_pretraining_lr": (float(spt_lr) if student_pretraining_steps > 0 else None),
        "student_pretraining_eval_interval": spt_eval_interval,
        "student_pretraining_eval_episodes": spt_eval_episodes,
        "student_pretrain_eval_by_step": list(student_pretrain_eval),
        "student_sft_prep_final_loss": (
            float(student_pretrain_losses[-1]) if student_pretrain_losses else None
        ),
    }
    save_results_csv(run_dir, results)

    plot_losses_pdf(
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
        bc_param_delta_l2_by_iter=bc_param_delta_l2_by_iter or None,
        bc_param_delta_iters=bc_param_delta_iters or None,
    )
    print(
        f"\nFinal write of losses PDF to {losses_pdf} "
        f"(also updated during SFT prep, each rollout episode, after BC, and after teacher eval; "
        f"page 1: teacher + optional SFT prep + BC + success + ||Δθ|| per BC phase; page 2+: per-BC-phase)"
    )
    print(
        f"Distillation rollout metrics (multi-page PDF per iter): "
        f"{video_dir}/iter_*/distillation_rollout_metrics.pdf"
    )
    print(f"Saved results to {run_dir / 'results.csv'} (includes teacher_eval_success_rate)")

    if wb_run is not None:
        wandb.summary["teacher_eval_success_rate"] = float(teacher_eval_sr)
        wandb.summary["stopped_reason"] = stopped_reason
        wandb.summary["final_success"] = bool(final_success)
        wandb.summary["n_distill_outer_iters_logged"] = len(distill_eval_by_iter)
        if student_final_eval_10ep_sr is not None:
            wandb.summary["student_final_eval_10ep_success_rate"] = float(
                student_final_eval_10ep_sr
            )
        wandb.finish()

