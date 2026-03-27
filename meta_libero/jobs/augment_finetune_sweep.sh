#!/bin/bash
# Run augment_finetune_experiment for task pairs with and without augmentation.
# Submits SLURM jobs for each combination.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOG_DIR="${META_LIBERO_LOG_DIR:-${PROJECT_ROOT}/meta_libero/logs}"
VENV_PATH="${META_VENV_PATH:-${PROJECT_ROOT}/.venv}"

mkdir -p "${LOG_DIR}"

# ============== TASK PAIRS & AUGMENTATION ==============
TASK_PAIRS=(
  "1 3"
  "3 1"
  "4 5"
  "5 4"
  "6 7"
  "7 6"
)
# Phase 2 mode: "self_replay", "two_lrs" (self-replay + dual LR), "cotraining", "on_policy_self_replay", "self_check", "no_augment"
MODES=("on_policy_self_replay")   # e.g. "two_lrs" "self_replay" "cotraining"
SINGLE_EPISODE_OPTS=("")   # ("" "--single_episode")
# Finetune type: "" (full), "lora", or "action_expert_only"
FINETUNE_TYPES=("")  # ("" "lora" "action_expert_only")


# Hyperparameters to sweep
LEARNING_RATES=(1e-04)
NUM_STEPS_LIST=(200)
MERGING_EPS_LIST=(1 0.75)  # 1.0=keep phase2, 0=keep phase1, 0.5=50/50 blend
# Alignment ratio threshold for self_replay/on_policy_self_replay: "" = disable, "0.2" = keep samples with ratio <= 0.2
ALIGNMENT_RATIO_THRESHOLD_LIST=("") # ("" "0.2")
# LR warmup: 0 = no warmup (default), 20 = linear 0->lr over 20 steps then constant
WARMUP_STEPS_LIST=(0)
# For self_check mode: weight for augmented samples (default 1)
LAMBDA_KL_LIST=(1)

# SLURM settings
TIME="4:00:00"
MEM="64G"
GPU="pro_6000:1"

# ============== JOB SUBMISSION ==============
echo "Submitting augment_finetune jobs..."
echo "=================================="
echo "Logs directory: ${LOG_DIR}"

job_count=0

for pair in "${TASK_PAIRS[@]}"; do
  read -r t1 t2 <<< "$pair"
  for LR in "${LEARNING_RATES[@]}"; do
    for NUM_STEPS in "${NUM_STEPS_LIST[@]}"; do
        for MERGING_EPS in "${MERGING_EPS_LIST[@]}"; do
        for ALIGN_THRESH in "${ALIGNMENT_RATIO_THRESHOLD_LIST[@]}"; do
        for WARMUP_STEPS in "${WARMUP_STEPS_LIST[@]}"; do
        for mode in "${MODES[@]}"; do
          if [[ "${mode}" == "self_check" ]]; then
            LAMBDA_KL_VALUES=("${LAMBDA_KL_LIST[@]}")
          else
            LAMBDA_KL_VALUES=(1)
          fi
          for LAMBDA_KL in "${LAMBDA_KL_VALUES[@]}"; do
          for single_episode in "${SINGLE_EPISODE_OPTS[@]}"; do
            for finetune_type in "${FINETUNE_TYPES[@]}"; do
              if [[ "${mode}" == "self_replay" ]]; then
                suffix=""
              elif [[ "${mode}" == "two_lrs" ]]; then
                suffix="_two_lrs"
              elif [[ "${mode}" == "no_augment" ]]; then
                suffix="_noaugment"
              elif [[ "${mode}" == "on_policy_self_replay" ]]; then
                suffix="_onpolicy_self_replay"
              elif [[ "${mode}" == "self_check" ]]; then
                suffix="_self_check"
              else
                suffix="_cotraining"
              fi
              if [[ -n "${single_episode}" ]]; then
                suffix="${suffix}_single"
              fi
              if [[ -n "${finetune_type}" ]]; then
                suffix="${suffix}_${finetune_type}"
              fi
              if [[ -n "${ALIGN_THRESH}" ]]; then
                suffix="${suffix}_align${ALIGN_THRESH}"
              fi
              if [[ "${WARMUP_STEPS}" -gt 0 ]]; then
                suffix="${suffix}_warmup${WARMUP_STEPS}"
              fi
              if [[ "${mode}" == "self_check" && "${LAMBDA_KL}" != "1" && "${LAMBDA_KL}" != "1.0" ]]; then
                suffix="${suffix}_lambdaKL${LAMBDA_KL}"
              fi
              JOB_NAME="augft_t${t1}_t${t2}_lr${LR}_st${NUM_STEPS}_eps${MERGING_EPS}${suffix}"

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

              WARMUP_FLAG=""
              if [[ "${WARMUP_STEPS}" -gt 0 ]]; then
                WARMUP_FLAG="--warmup_steps ${WARMUP_STEPS}"
              fi

              LAMBDA_KL_FLAG=""
              if [[ "${mode}" == "self_check" ]]; then
                LAMBDA_KL_FLAG="--lambda_kl ${LAMBDA_KL}"
              fi

              echo "Submitting: task1=${t1} task2=${t2} lr=${LR} steps=${NUM_STEPS} mode=${mode} merging_eps=${MERGING_EPS} align=${ALIGN_THRESH:-none} warmup=${WARMUP_STEPS} lambda_kl=${LAMBDA_KL} ${single_episode} ${finetune_type}"

              sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME}
#SBATCH --mem-per-cpu=${MEM}
#SBATCH --gpus=${GPU}
#SBATCH --output=${LOG_DIR}/augft_%j.out
#SBATCH --error=${LOG_DIR}/augft_%j.err

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"

python meta_libero/scripts/augment_finetune_experiment.py \
  --task1 ${t1} \
  --task2 ${t2} \
  --lr ${LR} \
  --num_steps ${NUM_STEPS} \
  --mode ${mode} \
  --merging_eps ${MERGING_EPS} \
  ${single_episode} \
  ${FINETUNE_FLAG} \
  ${ALIGN_FLAG} \
  ${WARMUP_FLAG} \
  ${LAMBDA_KL_FLAG}
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

echo "=================================="
echo "Submitted ${job_count} jobs total"
echo "Check status with: squeue -u \$USER"
