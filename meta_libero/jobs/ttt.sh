#!/bin/bash
#SBATCH --job-name=ttt_libero
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=a100-pcie-40gb:1
#SBATCH --output=/cluster/home/anmari/meta_vlas/meta_libero/logs/ttt_%j.out
#SBATCH --error=/cluster/home/anmari/meta_vlas/meta_libero/logs/ttt_%j.err


# Note: seed 0 has train=True in compute_loss (preprocessing observation)
# seed 1 has train=False in compute_loss (no preprocessing observation)
# seed 4,5,6 after the bugfix of policy.infer

# Alternative gpu: a100-pcie-40gb:1
# Or a100_80gb:1
cd /cluster/home/anmari/meta_vlas
source .venv/bin/activate

# TTT Evaluation Parameters
TASK_SUITE_NAME="libero_90"
TASK_ID=1
NUM_TRIALS=50 # 50
LR=2.5e-03
TTT_FREQUENCY=20
TTT_NUM_STEPS=10
TTT_K=6 # use 6
SEED=5
USE_BASE_MODEL=""  # Set to "--use-base-model" to enable
SAVE_VIDEO="--save-video"  # Set to "--save-video" to enable
ACTION_EXPERT_ONLY=""  # Set to "--action-expert-only" to enable
USE_LORA="--use-lora"  # Set to "--use-lora" to enable
NO_RESET_POLICY=""  # Set to "--no-reset-policy" to disable
USE_BASE_MODEL=""  # Set to "--use-base-model" to enable

python meta_libero/scripts/ttt_evaluation.py \
    --task_suite_name "$TASK_SUITE_NAME" \
    --task_id "$TASK_ID" \
    --num_trials "$NUM_TRIALS" \
    --lr "$LR" \
    --ttt_frequency "$TTT_FREQUENCY" \
    --ttt_num_steps "$TTT_NUM_STEPS" \
    --ttt_k "$TTT_K" \
    --seed "$SEED" \
    $USE_BASE_MODEL \
    $SAVE_VIDEO \
    $ACTION_EXPERT_ONLY \
    $USE_LORA \
    $NO_RESET_POLICY \
    $USE_BASE_MODEL
