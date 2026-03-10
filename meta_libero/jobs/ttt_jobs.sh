#!/bin/bash
# Script to submit multiple TTT evaluation jobs with different hyperparameters
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
LOG_DIR="${META_LIBERO_LOG_DIR:-${PROJECT_ROOT}/meta_libero/logs}"
VENV_PATH="${META_VENV_PATH:-${PROJECT_ROOT}/.venv}"

# Ensure SLURM output/error directory exists.
mkdir -p "${LOG_DIR}"

# ============== HYPERPARAMETER GRID ==============
# Modify these arrays to change the hyperparameter search

TASK_SUITE_NAME="libero_90"
TASK_IDS=(0)                     # Task IDs to evaluate
SEEDS=(1)                       # Seeds to iterate over
LEARNING_RATES=(2.5e-04)            # Learning rates
TTT_FREQUENCIES=(20)                # TTT frequency (every N steps)
TTT_NUM_STEPS_LIST=(5)           # Number of gradient steps per TTT update
TTT_K_VALUES=(6)                    # Number of nearest neighbors
MAX_TTT_STEPS=(1000)
META_UPDATES=("tt_reptile")              # Options: reset, continual_ttt, tt_reptile
MERGING_EPS_VALUES=(0.0 0.25 0.5 0.75 1.0)           # Used only when meta_update=tt_reptile
FINETUNE_TYPES=("lora" "action_expert_only")

# Fixed parameters
NUM_TRIALS=50
SAVE_VIDEO="--save-video"           # Set to "" to disable
USE_BASE_MODEL=""                   # Set to "--use-base-model" to enable
NO_RESET="--no_reset"                         # Set to "--no_reset" to disable reset across episodes
NO_MIRROR="--no-mirror-data"                        # Set to "--no-mirror-data" to disable mirrored dataloader transform

# ! Important: Use noise injection instead of TTT
NOISE_TTT=""             # Set to "" to disable
# ! Important: Use libero_90 dataset instead of libero_10 dataset for the TTT dataset
DATASET_TO_USE="--libero-90-dataset"  # Set to "--libero_90_dataset" to use libero_90 dataset

# SLURM settings
TIME="24:00:00"
MEM="64G"

# For interactive job: srun --time=4:0:0 --mem-per-cpu=32G --gpus= pro_6000:1--pty bash -l
# GPUs available: v100:1, a100-pcie-40gb:1, a100_80gb:1, pro_6000:1
GPU="pro_6000:1"

# ============== JOB SUBMISSION ==============
echo "Submitting TTT evaluation jobs..."
echo "=================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Logs directory: ${LOG_DIR}"

job_count=0

for TASK_ID in "${TASK_IDS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for LR in "${LEARNING_RATES[@]}"; do
            for TTT_FREQ in "${TTT_FREQUENCIES[@]}"; do
                for TTT_STEPS in "${TTT_NUM_STEPS_LIST[@]}"; do
                    for TTT_K in "${TTT_K_VALUES[@]}"; do
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

                            for META_UPDATE in "${META_UPDATES[@]}"; do
                            # Use a single submission template; only MERGING_EPS_FLAG changes.
                            if [[ "${META_UPDATE}" == "tt_reptile" ]]; then
                                EPS_CANDIDATES=("${MERGING_EPS_VALUES[@]}")
                            else
                                EPS_CANDIDATES=("")
                            fi

                            for MERGING_EPS in "${EPS_CANDIDATES[@]}"; do
                                MERGING_EPS_FLAG=""
                                JOB_SUFFIX="${META_UPDATE}"
                                if [[ "${META_UPDATE}" == "tt_reptile" ]]; then
                                    MERGING_EPS_FLAG="--merging_eps ${MERGING_EPS}"
                                    JOB_SUFFIX="${JOB_SUFFIX}_eps${MERGING_EPS}"
                                fi

                                JOB_NAME="ttt_${FINETUNE_TYPE}_t${TASK_ID}_s${SEED}_lr${LR}_f${TTT_FREQ}_st${TTT_STEPS}_k${TTT_K}_${JOB_SUFFIX}"
                                echo "Submitting: type=$FINETUNE_TYPE, task=$TASK_ID, seed=$SEED, lr=$LR, freq=$TTT_FREQ, steps=$TTT_STEPS, k=$TTT_K, meta_update=$META_UPDATE ${MERGING_EPS_FLAG}"

                                sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME}
#SBATCH --mem-per-cpu=${MEM}
#SBATCH --gpus=${GPU}
#SBATCH --output=${LOG_DIR}/ttt_%j.out
#SBATCH --error=${LOG_DIR}/ttt_%j.err

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"

python meta_libero/scripts/ttt_evaluation.py \
    --task_suite_name "${TASK_SUITE_NAME}" \
    --task_id ${TASK_ID} \
    --num_trials ${NUM_TRIALS} \
    --lr ${LR} \
    --ttt_frequency ${TTT_FREQ} \
    --ttt_num_steps ${TTT_STEPS} \
    --ttt_k ${TTT_K} \
    --meta_update ${META_UPDATE} \
    ${MERGING_EPS_FLAG} \
    ${NO_RESET} \
    --seed ${SEED} \
    --max_ttt_step ${MAX_TTT_STEPS} \
    ${MODEL_FLAG} \
    ${SAVE_VIDEO} \
    ${USE_BASE_MODEL} \
    ${NOISE_TTT} \
    ${DATASET_TO_USE} \
    ${NO_MIRROR}
EOF

                                job_count=$((job_count + 1))
                            done

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
