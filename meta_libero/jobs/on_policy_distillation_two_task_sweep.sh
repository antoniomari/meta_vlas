#!/bin/bash
# Submit SLURM jobs for on_policy_distillation_two_task.py (sequential task1 -> task2, student carries over).
# Same layout as on_policy_distillation_sweep.sh (PROJECT_ROOT, venv, logs).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOG_DIR="${META_LIBERO_LOG_DIR:-${PROJECT_ROOT}/meta_libero/logs}"
VENV_PATH="${META_VENV_PATH:-${PROJECT_ROOT}/.venv}"

mkdir -p "${LOG_DIR}"

# ============== TASK PAIRS ==============
# Phase 1: --task (task1); phase 2: --task2. Skip pairs where task1 == task2.
TASKS=(7)
TASK2S=(0)

# Optional sweeps
SINGLE_EPISODE_OPTS=("")
# "" = standard full-model fine-tuning; "lora" -> --lora; "action_expert_only" -> --action_expert_only (mutually exclusive)
FINETUNE_TYPES=("lora" "")
# "" = each BC phase uses only the latest rollout; add "--cumulative_buffer" to accumulate data across outer iterations
CUMULATIVE_OPTS=("")
SAVE_VIDEO_OPTS=("")
# "" = early stop on rollout success (default); "--full_experiment" = all max_iters / max_iters_task2, periodic 10-ep on task1 (phase 1)
FULL_EXPERIMENT_OPTS=("--full_experiment")

# ============== HYPERPARAMETERS ==============
TEACHER_LRS=(1e-04)
TEACHER_STEPS_LIST=(200)
# If empty string, omit --teacher_steps_task2 (defaults to --teacher_steps)
TEACHER_STEPS_TASK2_LIST=("")
BC_LRS=(2.5e-05)
BC_STEPS_LIST=(20)
# If empty, omit --bc_steps_task2 / --bc_lr_task2 (defaults match task1)
BC_STEPS_TASK2_LIST=(20)
BC_LR_TASK2_LIST=("")
MAX_ITERS_LIST=(20)
MAX_ITERS_TASK2_LIST=(20)
SEEDS=(42)
TEACHER_EVAL_EPISODES_LIST=(10)
ROLLOUT_EPISODES_LIST=(1)
BATCH_SIZE=32
# Phase 2: dual 10-ep student eval (task1+task2) every N outer iterations (default 5)
PHASE2_EVAL_INTERVAL_LIST=(5)
# Distillation BC targets: (1-α)*teacher + α*student per replan; 0 = teacher-only (default / matches omitting _sam in run dir)
STUDENT_ACTION_MERGE_LIST=(0)
# "" = no phase-2 dataset self-replay; "--phase2_self_replay" = augment-style paired BC after each rollout BC (run dir _p2sr)
PHASE2_SELF_REPLAY_OPTS=("")

# Alignment (mutually exclusive per job)
ALIGNMENT_RATIO_THRESHOLD_LIST=("")
ALIGN_MIN_LIST=("")

# Per-replan distillation BC weight = temporal_decay ** env_step; 1.0 = uniform (default)
TEMPORAL_DECAY_LIST=(1.0)
# "" = L2 MSE on diffusion residual (default); "--l1_bc_loss" = L1 (student BC only)
L1_BC_LOSS_OPTS=("")
# Student BC: MSE to frozen ref student, weight kl_lambda (0 = off; run dir _kl… when non-zero)
KL_LAMBDA_LIST=(0.01 0.1 1.0)
# Student: G independent action-chunk samples per replan (1 = default; run dir _gG when G!=1)
GROUP_SIZE_LIST=(1)
# Teacher: T independent noise draws per replan (1 = default; run dir _tgT when T!=1)
TEACHER_GROUP_SIZE_LIST=(1)

# SLURM
TIME="4:00:00"
MEM="64G"
GPU="pro_6000:1"

# ============== JOB SUBMISSION ==============
echo "Submitting on_policy_distillation_two_task jobs..."
echo "=================================="
echo "Logs directory: ${LOG_DIR}"

job_count=0

for TASK in "${TASKS[@]}"; do
  for TASK2 in "${TASK2S[@]}"; do
    if [[ "${TASK}" == "${TASK2}" ]]; then
      continue
    fi
    for TEACHER_LR in "${TEACHER_LRS[@]}"; do
      for TEACHER_STEPS in "${TEACHER_STEPS_LIST[@]}"; do
        for TEACHER_STEPS_TASK2 in "${TEACHER_STEPS_TASK2_LIST[@]}"; do
          for BC_LR in "${BC_LRS[@]}"; do
            for BC_STEPS in "${BC_STEPS_LIST[@]}"; do
              for BC_STEPS_TASK2 in "${BC_STEPS_TASK2_LIST[@]}"; do
                for BC_LR_TASK2 in "${BC_LR_TASK2_LIST[@]}"; do
                  for MAX_ITERS in "${MAX_ITERS_LIST[@]}"; do
                    for MAX_ITERS_TASK2 in "${MAX_ITERS_TASK2_LIST[@]}"; do
                      for SEED in "${SEEDS[@]}"; do
                        for TEACHER_EVAL_EP in "${TEACHER_EVAL_EPISODES_LIST[@]}"; do
                          for ROLLOUT_EP in "${ROLLOUT_EPISODES_LIST[@]}"; do
                            for PHASE2_EVAL_INT in "${PHASE2_EVAL_INTERVAL_LIST[@]}"; do
                            for SAM in "${STUDENT_ACTION_MERGE_LIST[@]}"; do
                            for P2_SR in "${PHASE2_SELF_REPLAY_OPTS[@]}"; do
                            for TEMPORAL_DECAY in "${TEMPORAL_DECAY_LIST[@]}"; do
                            for L1_BC in "${L1_BC_LOSS_OPTS[@]}"; do
                            for KL_LAMBDA in "${KL_LAMBDA_LIST[@]}"; do
                            for GROUP_SIZE in "${GROUP_SIZE_LIST[@]}"; do
                            for TEACHER_GROUP_SIZE in "${TEACHER_GROUP_SIZE_LIST[@]}"; do
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
                                        if [[ -n "${TEACHER_STEPS_TASK2}" ]]; then
                                          suffix="${suffix}_ts2${TEACHER_STEPS_TASK2}"
                                        fi
                                        if [[ -n "${BC_STEPS_TASK2}" ]]; then
                                          suffix="${suffix}_bs2${BC_STEPS_TASK2}"
                                        fi
                                        if [[ -n "${BC_LR_TASK2}" ]]; then
                                          suffix="${suffix}_bclr2${BC_LR_TASK2}"
                                        fi
                                        if [[ -n "${full_exp}" ]]; then
                                          suffix="${suffix}_full"
                                        fi
                                        if [[ "${PHASE2_EVAL_INT}" != "5" ]]; then
                                          suffix="${suffix}_p2ev${PHASE2_EVAL_INT}"
                                        fi
                                        if [[ "${SAM}" != "0" ]]; then
                                          suffix="${suffix}_sam${SAM}"
                                        fi
                                        if [[ -n "${P2_SR}" ]]; then
                                          suffix="${suffix}_p2sr"
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

                                        JOB_NAME="opdist2_t${TASK}_t2_${TASK2}_tlr${TEACHER_LR}_ts${TEACHER_STEPS}_bc${BC_LR}_bs${BC_STEPS}_mi${MAX_ITERS}_mi2${MAX_ITERS_TASK2}_s${SEED}${suffix}"

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

                                        TS2_FLAG=""
                                        if [[ -n "${TEACHER_STEPS_TASK2}" ]]; then
                                          TS2_FLAG="--teacher_steps_task2 ${TEACHER_STEPS_TASK2}"
                                        fi
                                        BS2_FLAG=""
                                        if [[ -n "${BC_STEPS_TASK2}" ]]; then
                                          BS2_FLAG="--bc_steps_task2 ${BC_STEPS_TASK2}"
                                        fi
                                        BCLR2_FLAG=""
                                        if [[ -n "${BC_LR_TASK2}" ]]; then
                                          BCLR2_FLAG="--bc_lr_task2 ${BC_LR_TASK2}"
                                        fi

                                        echo "Submitting opdist2: task=${TASK} task2=${TASK2} seed=${SEED} batch=${BATCH_SIZE} tlr=${TEACHER_LR} ts=${TEACHER_STEPS} ts2=${TEACHER_STEPS_TASK2:-<default>} bclr=${BC_LR} bs=${BC_STEPS} bs2=${BC_STEPS_TASK2:-<default>} bclr2=${BC_LR_TASK2:-<default>} mi1=${MAX_ITERS} mi2=${MAX_ITERS_TASK2} p2ev=${PHASE2_EVAL_INT} sam=${SAM} p2sr=${P2_SR:-off} td=${TEMPORAL_DECAY} l1=${L1_BC:-off} kl=${KL_LAMBDA} g=${GROUP_SIZE} tg=${TEACHER_GROUP_SIZE} teval=${TEACHER_EVAL_EP} rollout=${ROLLOUT_EP} align=${ALIGN_THRESH:-none} align_min=${ALIGN_MIN:-none} se=${single_episode:-} ft=${finetune_type:-} cum=${cumulative:-} vid=${save_video_opt:-} full=${full_exp:-none}"

                                        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name="${JOB_NAME}"
#SBATCH --time="${TIME}"
#SBATCH --mem-per-cpu="${MEM}"
#SBATCH --gpus="${GPU}"
#SBATCH --output="${LOG_DIR}/opdist2_%j.out"
#SBATCH --error="${LOG_DIR}/opdist2_%j.err"

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"

python meta_libero/scripts/on_policy_distillation_two_task.py \
  --task "${TASK}" \
  --task2 "${TASK2}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --teacher_lr "${TEACHER_LR}" \
  --teacher_steps "${TEACHER_STEPS}" \
  ${TS2_FLAG} \
  --bc_lr "${BC_LR}" \
  --bc_steps "${BC_STEPS}" \
  ${BS2_FLAG} \
  ${BCLR2_FLAG} \
  --max_iters "${MAX_ITERS}" \
  --max_iters_task2 "${MAX_ITERS_TASK2}" \
  --teacher_eval_episodes "${TEACHER_EVAL_EP}" \
  --rollout_episodes "${ROLLOUT_EP}" \
  --phase2_eval_interval "${PHASE2_EVAL_INT}" \
  --student_action_merge "${SAM}" \
  ${single_episode} \
  ${FINETUNE_FLAG} \
  ${ALIGN_FLAG} \
  ${ALIGN_MIN_FLAG} \
  ${cumulative} \
  ${save_video_opt} \
  ${full_exp} \
  ${P2_SR} \
  --temporal_decay "${TEMPORAL_DECAY}" \
  --kl_lambda "${KL_LAMBDA}" \
  --group_size "${GROUP_SIZE}" \
  --teacher_group_size "${TEACHER_GROUP_SIZE}" \
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

echo "=================================="
echo "Submitted ${job_count} jobs total"
echo "Check status with: squeue -u \$USER"
