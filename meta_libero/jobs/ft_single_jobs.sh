#!/bin/bash
# Script to submit multiple fine-tuning jobs with different hyperparameters

# ============== HYPERPARAMETER GRID ==============
# Modify these arrays to change the hyperparameter search

TASK_SUITE_NAME="libero_90"
TASK_IDS=(0 1 2 3 4 5 6 7)                        # Task IDs to fine-tune on
SEEDS=(1 2 3)                           # Seeds to iterate over
LEARNING_RATES=(2.5e-05)            # Learning rates
BATCH_SIZES=(32)                    # Batch sizes
TOTAL_STEPS=(500)                   # Total gradient steps
EVAL_INTERVALS=(100)                 # Evaluate every N steps
WARMUP_STEPS=(0)                 # LR warmup steps

# Fixed parameters
NUM_TRIALS=50
SAVE_VIDEO="--save-video"                       # Set to "--save-video" to enable
USE_LORA=""                         # Set to "--use-lora" to enable
ACTION_EXPERT_ONLY="--action-expert-only"               # Set to "--action-expert-only" to enable
USE_BASE_MODEL=""                   # Set to "--use-base-model" to enable
SKIP_FIRST_EVAL=""                  # Set to "--skip-first-eval" to skip step-0 eval
DATASET_TO_USE="--libero-90-dataset"                   # Set to "--libero_90_dataset" to use libero_90

# SLURM settings
TIME="24:00:00"
MEM="64G"

# GPUs available: v100:1, a100-pcie-40gb:1
GPU="a100_80gb:1"
LOG_DIR="/cluster/home/anmari/meta_vlas/meta_libero/logs"

# ============== JOB SUBMISSION ==============
echo "Submitting fine-tuning jobs..."
echo "=================================="

job_count=0

for TASK_ID in "${TASK_IDS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for LR in "${LEARNING_RATES[@]}"; do
            for BS in "${BATCH_SIZES[@]}"; do
                for STEPS in "${TOTAL_STEPS[@]}"; do
                    for EVAL_INT in "${EVAL_INTERVALS[@]}"; do
                        for WARMUP in "${WARMUP_STEPS[@]}"; do

                            JOB_NAME="ft_t${TASK_ID}_s${SEED}_lr${LR}_b${BS}_st${STEPS}"

                            echo "Submitting: task=$TASK_ID, seed=$SEED, lr=$LR, batch=$BS, steps=$STEPS, eval_int=$EVAL_INT"

                            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME}
#SBATCH --mem-per-cpu=${MEM}
#SBATCH --gpus=${GPU}
#SBATCH --output=${LOG_DIR}/ft_%j.out
#SBATCH --error=${LOG_DIR}/ft_%j.err

cd /cluster/home/anmari/meta_vlas
source .venv/bin/activate

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
    ${USE_LORA} \
    ${SAVE_VIDEO} \
    ${ACTION_EXPERT_ONLY} \
    ${USE_BASE_MODEL} \
    ${SKIP_FIRST_EVAL} \
    ${DATASET_TO_USE}
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

echo "=================================="
echo "Submitted ${job_count} jobs total"
echo "Check status with: squeue -u \$USER"
