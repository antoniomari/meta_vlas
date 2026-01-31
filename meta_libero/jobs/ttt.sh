#!/bin/bash
#SBATCH --job-name=ttt_libero
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=a100-pcie-40gb:1
#SBATCH --output=/cluster/home/anmari/meta_vlas/meta_libero/logs/ttt_%j.out
#SBATCH --error=/cluster/home/anmari/meta_vlas/meta_libero/logs/ttt_%j.err


# Alternative gpu: a100-pcie-40gb:1
# Or a100_80gb:1
cd /cluster/home/anmari/meta_vlas
source .venv/bin/activate

# TTT Evaluation Parameters
TASK_SUITE_NAME="libero_10"
TASK_ID=8
NUM_TRIALS=50
LR=2.5e-5
TTT_FREQUENCY=50
TTT_NUM_STEPS=1
TTT_K=8
SEED=0
USE_BASE_MODEL=""  # Set to "--use-base-model" to enable
SAVE_VIDEO=""  # Set to "--save-video" to enable
ACTION_EXPERT_ONLY=""  # Set to "--action-expert-only" to enable
USE_LORA="--use-lora"  # Set to "--use-lora" to enable

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
    $USE_LORA
