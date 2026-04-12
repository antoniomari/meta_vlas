"""Write results.csv for single-task on-policy distillation."""

import csv
from pathlib import Path


def save_results_csv(run_dir: Path, results: dict) -> None:
    """Augment-style metric/value CSV plus one row per distillation-iter success rate."""
    csv_path = run_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["task_id", results.get("task_id")])
        writer.writerow(["seed", results.get("seed")])
        writer.writerow(["rollout_episodes_per_iter", results.get("rollout_episodes")])
        writer.writerow(["rollout_num_envs", results.get("rollout_num_envs")])
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
        writer.writerow(["n_epoch", results.get("n_epoch")])
        writer.writerow(["bc_grad_steps_last_iter", results.get("bc_grad_steps_last_iter")])
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
        writer.writerow(["grpo_trust_eps", results.get("grpo_trust_eps")])
        writer.writerow(["grpo_weight", results.get("grpo_weight")])
        writer.writerow(["grpo_weight_eps", results.get("grpo_weight_eps")])
        writer.writerow(["distill_collect_every", results.get("distill_collect_every")])
        writer.writerow(
            [
                "save_distillation_rollout_metrics_pdf",
                results.get("save_distillation_rollout_metrics_pdf"),
            ]
        )
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
