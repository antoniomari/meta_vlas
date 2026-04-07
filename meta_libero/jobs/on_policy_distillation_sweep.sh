#!/bin/bash
# Submit SLURM jobs for on_policy_distillation.py over tasks and hyperparameters.
# Same layout style as augment_finetune_sweep.sh (PROJECT_ROOT, venv, logs).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOG_DIR="${META_LIBERO_LOG_DIR:-${PROJECT_ROOT}/meta_libero/logs}"
VENV_PATH="${META_VENV_PATH:-${PROJECT_ROOT}/.venv}"

mkdir -p "${LOG_DIR}"

# ============== TASKS & OPTIONS ==============
# Single task id per job (teacher FT + student distillation on that task)
TASKS=(6 7)

# Optional sweeps (mirror augment_finetune_sweep.sh: add entries to sweep)
SINGLE_EPISODE_OPTS=("")
FINETUNE_TYPES=("lora")
# "" = last-episode BC only; add "--cumulative_buffer" to sweep cumulative buffer
CUMULATIVE_OPTS=("")
# "" = save videos; add "--no_save_video" for lighter jobs
SAVE_VIDEO_OPTS=("")
# "" = early stop on rollout success (default); "--full_experiment" = run all max_iters, periodic 10-ep eval
FULL_EXPERIMENT_OPTS=("--full_experiment")

# ============== HYPERPARAMETERS ==============
TEACHER_LRS=(1e-04)
TEACHER_STEPS_LIST=(200)
BC_LRS=(5e-05)
BC_STEPS_LIST=(20)
MAX_ITERS_LIST=(100)
SEEDS=(42)
TEACHER_EVAL_EPISODES_LIST=(10)
# Episodes per distillation outer iter (merged into one BC dataset); 1 = previous behavior
ROLLOUT_EPISODES_LIST=(1)
# Distillation BC targets: (1-α)*teacher + α*student per replan; 0 = teacher-only (default)
STUDENT_ACTION_MERGE_LIST=(0)
BATCH_SIZE=32
# Alignment filter (mutually exclusive per job — do not set both in the same combination):
# "" = no alignment filter for that axis.
# ALIGNMENT_RATIO_THRESHOLD_LIST: "0.5" -> --alignment_ratio_threshold 0.5 (run dir _align0.5)
ALIGNMENT_RATIO_THRESHOLD_LIST=("")
# ALIGN_MIN_LIST: "0.15" -> --align_min 0.15 (keep ratio >= value; run dir _alignmin0.15)
ALIGN_MIN_LIST=("")

# Per-replan distillation BC weight = temporal_decay ** env_step; 1.0 = uniform (default)
TEMPORAL_DECAY_LIST=(1)
# "" = L2 MSE on diffusion residual (default); "--l1_bc_loss" = L1 (student BC only)
L1_BC_LOSS_OPTS=("--l1_bc_loss" "")
# Student BC: weight on MSE(v_t, stop_grad(v_t^ref)) vs frozen snapshot (0 = off; run dir _kl… when non-zero)
KL_LAMBDA_LIST=(0)
# Student: G independent action-chunk samples per replan (1 = default; run dir _gG when G!=1)
GROUP_SIZE_LIST=(4)
# Teacher: T independent noise draws per replan for variance in distillation_rollout_metrics.pdf (1 = default / legacy; run dir _tgT when T!=1)
TEACHER_GROUP_SIZE_LIST=(1)
# "" = no filter; else --max_teacher_variance (drop BC rows with mean_{h,d}Var(teacher samples) > this; run dir _mtv…)
MAX_TEACHER_VARIANCE_LIST=("")

# Before OPD: student offline SFT on a separate single-episode demo (same task). 0 = skip (run dir has no _spts).
STUDENT_PRETRAINING_STEPS_LIST=(0)
# "" = omit --student_pretraining_lr (Python default: same as --teacher_lr); else e.g. "2.5e-05"
STUDENT_PRETRAINING_LR_OPTS=("2.5e-05")
STUDENT_PRETRAINING_EVAL_INTERVAL_LIST=(50)
STUDENT_PRETRAINING_EVAL_EPISODES_LIST=(10)

# "" = standard distillation BC; "--grpo_like" = advantage-weighted diffusion loss (requires GROUP_SIZE>=2)
GRPO_LIKE_OPTS=("--grpo_like")
# GRPO only: "" = omit (raw student chunk as BC target); else e.g. "0.05" -> --grpo_trust_eps (run dir …_gte…)
GRPO_TRUST_EPS_OPTS=("0.1")
# GRPO only: "" = omit (--grpo_weight none); "mean_std" -> --grpo_weight mean_std (run dir …_gwms… from Python)
GRPO_WEIGHT_OPTS=("" "mean_std")
# Optional: set e.g. "1e-7" to pass --grpo_weight_eps (default in Python: 1e-8); leave empty to omit
GRPO_WEIGHT_EPS=""
# Distillation: BC rows every N env steps after --num_steps_wait (default 1; run dir …_dceN… when N!=1)
DISTILL_COLLECT_EVERY=5

# SLURM settings (match augment_finetune_sweep.sh)
TIME="24:00:00"
MEM="64G"
GPU="pro_6000:1"

# ============== JOB SUBMISSION ==============
echo "Submitting on_policy_distillation jobs..."
echo "=================================="
echo "Logs directory: ${LOG_DIR}"

job_count=0

for TASK in "${TASKS[@]}"; do
  for TEACHER_LR in "${TEACHER_LRS[@]}"; do
    for TEACHER_STEPS in "${TEACHER_STEPS_LIST[@]}"; do
      for BC_LR in "${BC_LRS[@]}"; do
        for BC_STEPS in "${BC_STEPS_LIST[@]}"; do
          for MAX_ITERS in "${MAX_ITERS_LIST[@]}"; do
            for SEED in "${SEEDS[@]}"; do
              for SPT_STEPS in "${STUDENT_PRETRAINING_STEPS_LIST[@]}"; do
              for SPT_LR_OPT in "${STUDENT_PRETRAINING_LR_OPTS[@]}"; do
              for SPT_EVAL_INT in "${STUDENT_PRETRAINING_EVAL_INTERVAL_LIST[@]}"; do
              for SPT_EVAL_EP in "${STUDENT_PRETRAINING_EVAL_EPISODES_LIST[@]}"; do
              for TEACHER_EVAL_EP in "${TEACHER_EVAL_EPISODES_LIST[@]}"; do
                for ROLLOUT_EP in "${ROLLOUT_EPISODES_LIST[@]}"; do
                for SAM in "${STUDENT_ACTION_MERGE_LIST[@]}"; do
                for TEMPORAL_DECAY in "${TEMPORAL_DECAY_LIST[@]}"; do
                for L1_BC in "${L1_BC_LOSS_OPTS[@]}"; do
                for KL_LAMBDA in "${KL_LAMBDA_LIST[@]}"; do
                for GROUP_SIZE in "${GROUP_SIZE_LIST[@]}"; do
                for TEACHER_GROUP_SIZE in "${TEACHER_GROUP_SIZE_LIST[@]}"; do
                for MAX_TEACHER_VARIANCE in "${MAX_TEACHER_VARIANCE_LIST[@]}"; do
                for ALIGN_THRESH in "${ALIGNMENT_RATIO_THRESHOLD_LIST[@]}"; do
                for ALIGN_MIN in "${ALIGN_MIN_LIST[@]}"; do
                  if [[ -n "${ALIGN_THRESH}" && -n "${ALIGN_MIN}" ]]; then
                    continue
                  fi
                  for single_episode in "${SINGLE_EPISODE_OPTS[@]}"; do
                    for finetune_type in "${FINETUNE_TYPES[@]}"; do
                      for cumulative in "${CUMULATIVE_OPTS[@]}"; do
                        for save_video_opt in "${SAVE_VIDEO_OPTS[@]}"; do
                        for full_exp in "${FULL_EXPERIMENT_OPTS[@]}"; do
                        for GRPO_OPT in "${GRPO_LIKE_OPTS[@]}"; do
                        for GRPO_TRUST_EPS in "${GRPO_TRUST_EPS_OPTS[@]}"; do
                        for GRPO_WEIGHT in "${GRPO_WEIGHT_OPTS[@]}"; do
                          if [[ -n "${GRPO_OPT}" && "${GROUP_SIZE}" -lt 2 ]]; then
                            continue
                          fi
                          if [[ -n "${GRPO_TRUST_EPS}" && -z "${GRPO_OPT}" ]]; then
                            continue
                          fi
                          if [[ -n "${GRPO_WEIGHT}" && -z "${GRPO_OPT}" ]]; then
                            continue
                          fi
                          if [[ -n "${GRPO_WEIGHT_EPS}" && -z "${GRPO_OPT}" ]]; then
                            continue
                          fi
                        suffix=""
                          if [[ -n "${single_episode}" ]]; then
                            suffix="${suffix}_single"
                          fi
                          if [[ -n "${finetune_type}" ]]; then
                            suffix="${suffix}_${finetune_type}"
                          fi
                          if [[ -n "${cumulative}" ]]; then
                            suffix="${suffix}_cumulative"
                          fi
                          if [[ -n "${save_video_opt}" ]]; then
                            suffix="${suffix}_novid"
                          fi
                          if [[ -n "${ALIGN_THRESH}" ]]; then
                            suffix="${suffix}_align${ALIGN_THRESH}"
                          fi
                          if [[ -n "${ALIGN_MIN}" ]]; then
                            suffix="${suffix}_alignmin${ALIGN_MIN}"
                          fi
                          if [[ "${ROLLOUT_EP}" != "1" ]]; then
                            suffix="${suffix}_rep${ROLLOUT_EP}"
                          fi
                          if [[ -n "${full_exp}" ]]; then
                            suffix="${suffix}_full"
                          fi
                          if [[ "${SAM}" != "0" ]]; then
                            suffix="${suffix}_sam${SAM}"
                          fi
                          if [[ "${TEMPORAL_DECAY}" != "1" && "${TEMPORAL_DECAY}" != "1.0" ]]; then
                            suffix="${suffix}_td${TEMPORAL_DECAY}"
                          fi
                          if [[ -n "${L1_BC}" ]]; then
                            suffix="${suffix}_l1bc"
                          fi
                          if [[ "${KL_LAMBDA}" != "0" && "${KL_LAMBDA}" != "0.0" ]]; then
                            suffix="${suffix}_kl${KL_LAMBDA}"
                          fi
                          if [[ "${GROUP_SIZE}" != "1" ]]; then
                            suffix="${suffix}_g${GROUP_SIZE}"
                          fi
                          if [[ "${TEACHER_GROUP_SIZE}" != "1" ]]; then
                            suffix="${suffix}_tg${TEACHER_GROUP_SIZE}"
                          fi
                          if [[ -n "${MAX_TEACHER_VARIANCE}" ]]; then
                            suffix="${suffix}_mtv${MAX_TEACHER_VARIANCE}"
                          fi
                          if [[ "${SPT_STEPS}" != "100" ]]; then
                            suffix="${suffix}_spts${SPT_STEPS}"
                          fi
                          if [[ -n "${SPT_LR_OPT}" ]]; then
                            suffix="${suffix}_sptlr${SPT_LR_OPT}"
                          fi
                          if [[ "${SPT_EVAL_INT}" != "50" ]]; then
                            suffix="${suffix}_sptevi${SPT_EVAL_INT}"
                          fi
                          if [[ "${SPT_EVAL_EP}" != "10" ]]; then
                            suffix="${suffix}_spte${SPT_EVAL_EP}"
                          fi
                          if [[ -n "${GRPO_OPT}" ]]; then
                            suffix="${suffix}_grpo"
                          fi
                          if [[ -n "${GRPO_TRUST_EPS}" ]]; then
                            suffix="${suffix}_gte${GRPO_TRUST_EPS}"
                          fi
                          if [[ "${GRPO_WEIGHT}" == "mean_std" ]]; then
                            suffix="${suffix}_gwms"
                          fi
                          if [[ -n "${GRPO_WEIGHT_EPS}" ]]; then
                            suffix="${suffix}_gwe${GRPO_WEIGHT_EPS}"
                          fi
                          if [[ "${DISTILL_COLLECT_EVERY}" != "1" ]]; then
                            suffix="${suffix}_dce${DISTILL_COLLECT_EVERY}"
                          fi

                          JOB_NAME="opdist_t${TASK}_tlr${TEACHER_LR}_ts${TEACHER_STEPS}_bc${BC_LR}_bs${BC_STEPS}_mi${MAX_ITERS}_s${SEED}${suffix}"

                          FINETUNE_FLAG=""
                          if [[ "${finetune_type}" == "lora" ]]; then
                            FINETUNE_FLAG="--lora"
                          elif [[ "${finetune_type}" == "action_expert_only" ]]; then
                            FINETUNE_FLAG="--action_expert_only"
                          fi

                          ALIGN_FLAG=""
                          if [[ -n "${ALIGN_THRESH}" ]]; then
                            ALIGN_FLAG="--alignment_ratio_threshold ${ALIGN_THRESH}"
                          fi
                          ALIGN_MIN_FLAG=""
                          if [[ -n "${ALIGN_MIN}" ]]; then
                            ALIGN_MIN_FLAG="--align_min ${ALIGN_MIN}"
                          fi
                          MAX_TV_FLAG=""
                          if [[ -n "${MAX_TEACHER_VARIANCE}" ]]; then
                            MAX_TV_FLAG="--max_teacher_variance ${MAX_TEACHER_VARIANCE}"
                          fi
                          SPT_LR_FLAG=""
                          if [[ -n "${SPT_LR_OPT}" ]]; then
                            SPT_LR_FLAG="--student_pretraining_lr ${SPT_LR_OPT}"
                          fi
                          TRUST_FLAG=""
                          if [[ -n "${GRPO_TRUST_EPS}" ]]; then
                            TRUST_FLAG="--grpo_trust_eps ${GRPO_TRUST_EPS}"
                          fi
                          GRPO_WEIGHT_FLAG=""
                          if [[ -n "${GRPO_WEIGHT}" ]]; then
                            GRPO_WEIGHT_FLAG="--grpo_weight ${GRPO_WEIGHT}"
                          fi
                          GRPO_WEIGHT_EPS_FLAG=""
                          if [[ -n "${GRPO_WEIGHT_EPS}" ]]; then
                            GRPO_WEIGHT_EPS_FLAG="--grpo_weight_eps ${GRPO_WEIGHT_EPS}"
                          fi

                          echo "Submitting: task=${TASK} teacher_lr=${TEACHER_LR} teacher_steps=${TEACHER_STEPS} bc_lr=${BC_LR} bc_steps=${BC_STEPS} max_iters=${MAX_ITERS} seed=${SEED} spt_steps=${SPT_STEPS} spt_lr=${SPT_LR_OPT:-teacher} spt_ev_int=${SPT_EVAL_INT} spt_ev_ep=${SPT_EVAL_EP} teval_ep=${TEACHER_EVAL_EP} rollout_ep=${ROLLOUT_EP} sam=${SAM} td=${TEMPORAL_DECAY} l1=${L1_BC:-off} kl=${KL_LAMBDA} g=${GROUP_SIZE} tg=${TEACHER_GROUP_SIZE} max_tv=${MAX_TEACHER_VARIANCE:-none} align=${ALIGN_THRESH:-none} align_min=${ALIGN_MIN:-none} full_exp=${full_exp:-none} grpo=${GRPO_OPT:-off} grpo_trust=${GRPO_TRUST_EPS:-none} grpo_w=${GRPO_WEIGHT:-none} grpo_w_eps=${GRPO_WEIGHT_EPS:-default} ${single_episode} ${finetune_type} ${cumulative} ${save_video_opt}"

                          sbatch <<EOF
#!/bin/bash
#SBATCH --job-name="${JOB_NAME}"
#SBATCH --time="${TIME}"
#SBATCH --mem-per-cpu="${MEM}"
#SBATCH --gpus="${GPU}"
#SBATCH --output="${LOG_DIR}/opdist_%j.out"
#SBATCH --error="${LOG_DIR}/opdist_%j.err"

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"

python meta_libero/scripts/on_policy_distillation.py \
  --task "${TASK}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --teacher_lr "${TEACHER_LR}" \
  --teacher_steps "${TEACHER_STEPS}" \
  --bc_lr "${BC_LR}" \
  --bc_steps "${BC_STEPS}" \
  --max_iters "${MAX_ITERS}" \
  --teacher_eval_episodes "${TEACHER_EVAL_EP}" \
  --rollout_episodes "${ROLLOUT_EP}" \
  --student_action_merge "${SAM}" \
  --group_size "${GROUP_SIZE}" \
  --teacher_group_size "${TEACHER_GROUP_SIZE}" \
  --temporal_decay "${TEMPORAL_DECAY}" \
  --kl_lambda "${KL_LAMBDA}" \
  ${MAX_TV_FLAG} \
  --student_pretraining_steps "${SPT_STEPS}" \
  ${SPT_LR_FLAG} \
  --student_pretraining_eval_interval "${SPT_EVAL_INT}" \
  --student_pretraining_eval_episodes "${SPT_EVAL_EP}" \
  ${single_episode} \
  ${FINETUNE_FLAG} \
  ${ALIGN_FLAG} \
  ${ALIGN_MIN_FLAG} \
  ${cumulative} \
  ${save_video_opt} \
  ${full_exp} \
  ${GRPO_OPT} \
  ${TRUST_FLAG} \
  ${GRPO_WEIGHT_FLAG} \
  ${GRPO_WEIGHT_EPS_FLAG} \
  --distill_collect_every "${DISTILL_COLLECT_EVERY}" \
  ${L1_BC}
EOF

                          job_count=$((job_count + 1))
                        done
                        done
                        done
                        done
                        done
                      done
                    done
                  done
                done
                done
                done
                done
                done
                done
                done
                done
                done
                done
              done
              done
              done
              done
              done
            done
          done
        done
      done
    done
  done
done

echo "=================================="
echo "Submitted ${job_count} jobs total"
echo "Check status with: squeue -u \$USER"
