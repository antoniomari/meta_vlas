#!/bin/bash
# Script to submit multiple fine-tuning jobs with different hyperparameters
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOG_DIR="${META_LIBERO_LOG_DIR:-${PROJECT_ROOT}/meta_libero/logs}"
VENV_PATH="${META_VENV_PATH:-${PROJECT_ROOT}/.venv}"

# Ensure SLURM output/error directory exists.
mkdir -p "${LOG_DIR}"

# ============== HYPERPARAMETER GRID ==============
# Modify these arrays to change the hyperparameter search

TASK_SUITE_NAME="libero_90"
TASK_IDS=(0)               # Task IDs to fine-tune on
SEEDS=(2 3)                           # Seeds to iterate over
LEARNING_RATES=(2.5e-05 2.5e-04)            # Learning rates
BATCH_SIZES=(32)                    # Batch sizes
TOTAL_STEPS=(500)                   # Total gradient steps
EVAL_INTERVALS=(100)                 # Evaluate every N steps
WARMUP_STEPS=(0)                 # LR warmup steps
FINETUNE_TYPES=("lora" "full" "action_expert_only")  # possible values: "lora", "full", "action_expert_only"

# Fixed parameters
NUM_TRIALS=50
SAVE_VIDEO="--save-video"                       # Set to "--save-video" to enable
USE_BASE_MODEL=""                   # Set to "--use-base-model" to enable
SKIP_FIRST_EVAL=""                  # Set to "--skip-first-eval" to skip step-0 eval
DATASET_TO_USE="--libero-90-dataset"                   # Set to "--libero_90_dataset" to use libero_90
NO_MIRROR=""                           # Set to "--no-mirror-data" to disable mirrored dataloader transform

# SLURM settings
TIME="24:00:00"
MEM="64G"

# GPUs available: a100_80gb:1, a100-pcie-40gb:1, pro_6000:1
GPU="pro_6000:1"

# ============== JOB SUBMISSION ==============
echo "Submitting fine-tuning jobs..."
echo "=================================="
echo "Logs directory: ${LOG_DIR}"

job_count=0

for TASK_ID in "${TASK_IDS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for LR in "${LEARNING_RATES[@]}"; do
            for BS in "${BATCH_SIZES[@]}"; do
                for STEPS in "${TOTAL_STEPS[@]}"; do
                    for EVAL_INT in "${EVAL_INTERVALS[@]}"; do
                        for WARMUP in "${WARMUP_STEPS[@]}"; do
                            for FINETUNE_TYPE in "${FINETUNE_TYPES[@]}"; do

                            MODEL_FLAG=""
                            if [[ "${FINETUNE_TYPE}" == "lora" ]]; then
                                MODEL_FLAG="--use-lora"
                            elif [[ "${FINETUNE_TYPE}" == "action_expert_only" ]]; then
                                MODEL_FLAG="--action-expert-only"
                            elif [[ "${FINETUNE_TYPE}" == "full" ]]; then
                                MODEL_FLAG=""
                            else
                                echo "Unknown FINETUNE_TYPE: ${FINETUNE_TYPE}"
                                exit 1
                            fi

                            JOB_NAME="ft_${FINETUNE_TYPE}_t${TASK_ID}_s${SEED}_lr${LR}_b${BS}_st${STEPS}"

                            echo "Submitting: type=$FINETUNE_TYPE, task=$TASK_ID, seed=$SEED, lr=$LR, batch=$BS, steps=$STEPS, eval_int=$EVAL_INT"

                            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME}
#SBATCH --mem-per-cpu=${MEM}
#SBATCH --gpus=${GPU}
#SBATCH --output=${LOG_DIR}/ft_%j.out
#SBATCH --error=${LOG_DIR}/ft_%j.err

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"

python meta_libero/scripts/finetune_single_task.py \
    --task_suite_name "${TASK_SUITE_NAME}" \
    --task_id ${TASK_ID} \
    --num_trials ${NUM_TRIALS} \
    --lr ${LR} \
    --batch_size ${BS} \
    --total_steps ${STEPS} \
    --eval_interval ${EVAL_INT} \
    --warmup_steps ${WARMUP} \
    --seed ${SEED} \
    ${MODEL_FLAG} \
    ${SAVE_VIDEO} \
    ${USE_BASE_MODEL} \
    ${SKIP_FIRST_EVAL} \
    ${DATASET_TO_USE} \
    ${NO_MIRROR}
EOF

                            job_count=$((job_count + 1))

                            # Optional: add delay between submissions to avoid overwhelming scheduler
                            # sleep 0.5

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
