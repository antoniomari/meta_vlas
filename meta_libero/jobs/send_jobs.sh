#!/bin/bash
# Script to submit multiple TTT evaluation jobs with different hyperparameters

# ============== HYPERPARAMETER GRID ==============
# Modify these arrays to change the hyperparameter search

TASK_SUITE_NAME="libero_90"
TASK_IDS=(0 1 2 3 4 5 6 7)                    # Task IDs to evaluate
SEEDS=(1 2 3)                       # Seeds to iterate over
LEARNING_RATES=(2.5e-04)            # Learning rates
TTT_FREQUENCIES=(5)                # TTT frequency (every N steps)
TTT_NUM_STEPS_LIST=(5)           # Number of gradient steps per TTT update
TTT_K_VALUES=(6)                    # Number of nearest neighbors
MAX_TTT_STEPS=(1000)

# Fixed parameters
NUM_TRIALS=50
USE_LORA="--use-lora"               # Set to "" to disable
SAVE_VIDEO="--save-video"           # Set to "" to disable
ACTION_EXPERT_ONLY=""               # Set to "--action-expert-only" to enable
NO_RESET_POLICY=""                  # Set to "--no-reset-policy" to enable
USE_BASE_MODEL=""                   # Set to "--use-base-model" to enable

# ! Important: Use noise injection instead of TTT
NOISE_TTT=""             # Set to "" to disable
# ! Important: Use libero_90 dataset instead of libero_10 dataset for the TTT dataset
DATASET_TO_USE="--libero-90-dataset"  # Set to "--libero_90_dataset" to use libero_90 dataset

# SLURM settings
TIME="24:00:00"
MEM="64G"

# GPUs available: v100:1, a100-pcie-40gb:1
GPU="a100-pcie-40gb:1"
LOG_DIR="/cluster/home/anmari/meta_vlas/meta_libero/logs"

# ============== JOB SUBMISSION ==============
echo "Submitting TTT evaluation jobs..."
echo "=================================="

job_count=0

for TASK_ID in "${TASK_IDS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for LR in "${LEARNING_RATES[@]}"; do
            for TTT_FREQ in "${TTT_FREQUENCIES[@]}"; do
                for TTT_STEPS in "${TTT_NUM_STEPS_LIST[@]}"; do
                    for TTT_K in "${TTT_K_VALUES[@]}"; do

                        JOB_NAME="ttt_t${TASK_ID}_s${SEED}_lr${LR}_f${TTT_FREQ}_st${TTT_STEPS}_k${TTT_K}"

                        echo "Submitting: task=$TASK_ID, seed=$SEED, lr=$LR, freq=$TTT_FREQ, steps=$TTT_STEPS, k=$TTT_K"

                        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME}
#SBATCH --mem-per-cpu=${MEM}
#SBATCH --gpus=${GPU}
#SBATCH --output=${LOG_DIR}/ttt_%j.out
#SBATCH --error=${LOG_DIR}/ttt_%j.err

cd /cluster/home/anmari/meta_vlas
source .venv/bin/activate

python meta_libero/scripts/ttt_evaluation.py \
    --task_suite_name "${TASK_SUITE_NAME}" \
    --task_id ${TASK_ID} \
    --num_trials ${NUM_TRIALS} \
    --lr ${LR} \
    --ttt_frequency ${TTT_FREQ} \
    --ttt_num_steps ${TTT_STEPS} \
    --ttt_k ${TTT_K} \
    --seed ${SEED} \
    --max_ttt_step ${MAX_TTT_STEPS} \
    ${USE_LORA} \
    ${SAVE_VIDEO} \
    ${ACTION_EXPERT_ONLY} \
    ${NO_RESET_POLICY} \
    ${USE_BASE_MODEL} \
    ${NOISE_TTT} \
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

echo "=================================="
echo "Submitted ${job_count} jobs total"
echo "Check status with: squeue -u \$USER"
