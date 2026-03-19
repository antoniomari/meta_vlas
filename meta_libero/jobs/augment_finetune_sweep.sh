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
# Phase 2 mode: "self_replay" (task_1 pseudo-labels), "cotraining" (task_2+task_1), "no_augment" (task_2 only)
MODES=("cotraining")   # ("self_replay" "cotraining" "no_augment")
SINGLE_EPISODE_OPTS=("")   # ("" "--single_episode")
# Finetune type: "" (full), "lora", or "action_expert_only"
FINETUNE_TYPES=("")  # ("" "lora" "action_expert_only")


# Hyperparameters to sweep
LEARNING_RATES=(5e-05 2.5e-05 2.5e-04)
NUM_STEPS_LIST=(100)

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
      for mode in "${MODES[@]}"; do
        for single_episode in "${SINGLE_EPISODE_OPTS[@]}"; do
          for finetune_type in "${FINETUNE_TYPES[@]}"; do
            if [[ "${mode}" == "self_replay" ]]; then
              suffix=""
            elif [[ "${mode}" == "no_augment" ]]; then
              suffix="_noaugment"
            else
              suffix="_cotraining"
            fi
            if [[ -n "${single_episode}" ]]; then
              suffix="${suffix}_single"
            fi
            if [[ -n "${finetune_type}" ]]; then
              suffix="${suffix}_${finetune_type}"
            fi
            JOB_NAME="augft_t${t1}_t${t2}_lr${LR}_st${NUM_STEPS}${suffix}"

            FINETUNE_FLAG=""
            if [[ "${finetune_type}" == "lora" ]]; then
              FINETUNE_FLAG="--lora"
            elif [[ "${finetune_type}" == "action_expert_only" ]]; then
              FINETUNE_FLAG="--action_expert_only"
            fi

            echo "Submitting: task1=${t1} task2=${t2} lr=${LR} steps=${NUM_STEPS} mode=${mode} ${single_episode} ${finetune_type}"

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
  ${single_episode} \
  ${FINETUNE_FLAG}
EOF

            job_count=$((job_count + 1))
          done
        done
      done
    done
  done
done

echo "=================================="
echo "Submitted ${job_count} jobs total"
echo "Check status with: squeue -u \$USER"
