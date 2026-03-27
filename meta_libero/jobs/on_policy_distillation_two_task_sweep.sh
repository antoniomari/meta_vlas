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
TASKS=(0)
TASK2S=(1 5 7)

# Optional sweeps
SINGLE_EPISODE_OPTS=("")
# "" = standard full-model fine-tuning; "lora" -> --lora; "action_expert_only" -> --action_expert_only (mutually exclusive)
FINETUNE_TYPES=("lora")
# "" = each BC phase uses only the latest rollout; add "--cumulative_buffer" to accumulate data across outer iterations
CUMULATIVE_OPTS=("")
SAVE_VIDEO_OPTS=("")

# ============== HYPERPARAMETERS ==============
TEACHER_LRS=(1e-04)
TEACHER_STEPS_LIST=(100)
# If empty string, omit --teacher_steps_task2 (defaults to --teacher_steps)
TEACHER_STEPS_TASK2_LIST=("")
BC_LRS=(2.5e-05)
BC_STEPS_LIST=(20)
# If empty, omit --bc_steps_task2 / --bc_lr_task2 (defaults match task1)
BC_STEPS_TASK2_LIST=(20)
BC_LR_TASK2_LIST=("")
MAX_ITERS_LIST=(5)
MAX_ITERS_TASK2_LIST=(25)
SEEDS=(42)
TEACHER_EVAL_EPISODES_LIST=(10)
ROLLOUT_EPISODES_LIST=(1)
BATCH_SIZE=32

# Alignment (mutually exclusive per job)
ALIGNMENT_RATIO_THRESHOLD_LIST=("")
ALIGN_MIN_LIST=("")

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
                            for ALIGN_THRESH in "${ALIGNMENT_RATIO_THRESHOLD_LIST[@]}"; do
                              for ALIGN_MIN in "${ALIGN_MIN_LIST[@]}"; do
                                if [[ -n "${ALIGN_THRESH}" && -n "${ALIGN_MIN}" ]]; then
                                  continue
                                fi
                                for single_episode in "${SINGLE_EPISODE_OPTS[@]}"; do
                                  for finetune_type in "${FINETUNE_TYPES[@]}"; do
                                    for cumulative in "${CUMULATIVE_OPTS[@]}"; do
                                      for save_video_opt in "${SAVE_VIDEO_OPTS[@]}"; do
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

                                        echo "Submitting: task=${TASK} task2=${TASK2} ... max_iters=${MAX_ITERS} max_iters_task2=${MAX_ITERS_TASK2} seed=${SEED}"

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
  ${single_episode} \
  ${FINETUNE_FLAG} \
  ${ALIGN_FLAG} \
  ${ALIGN_MIN_FLAG} \
  ${cumulative} \
  ${save_video_opt}
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

echo "=================================="
echo "Submitted ${job_count} jobs total"
echo "Check status with: squeue -u \$USER"
