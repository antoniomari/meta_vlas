#!/bin/bash
# Submit SLURM jobs for on_policy_distillation.py over tasks and hyperparameters.
# Same layout style as augment_finetune_sweep.sh (PROJECT_ROOT, venv, logs).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOG_DIR="${META_LIBERO_LOG_DIR:-${PROJECT_ROOT}/meta_libero/logs}"
VENV_PATH="${META_VENV_PATH:-${PROJECT_ROOT}/.venv}"

mkdir -p "${LOG_DIR}"

# ============== TASKS & OPTIONS ==============
# Single task id per job (teacher FT + student distillation on that task)
TASKS=(1 3 4 5 6)

# Optional sweeps (mirror augment_finetune_sweep.sh: add entries to sweep)
SINGLE_EPISODE_OPTS=("")
FINETUNE_TYPES=("")
# "" = last-episode BC only; add "--cumulative_buffer" to sweep cumulative buffer
CUMULATIVE_OPTS=("")
# "" = save videos; add "--no_save_video" for lighter jobs
SAVE_VIDEO_OPTS=("")
# "" = early stop on rollout success (default); "--full_experiment" = run all max_iters, periodic 10-ep eval
FULL_EXPERIMENT_OPTS=("--full_experiment")

# ============== HYPERPARAMETERS ==============
TEACHER_LRS=(1e-04)
TEACHER_STEPS_LIST=(100)
BC_LRS=(2.5e-05)
BC_STEPS_LIST=(10)
MAX_ITERS_LIST=(100)
SEEDS=(42)
TEACHER_EVAL_EPISODES_LIST=(10)
# Episodes per distillation outer iter (merged into one BC dataset); 1 = previous behavior
ROLLOUT_EPISODES_LIST=(1)
BATCH_SIZE=32
# Alignment filter (mutually exclusive per job — do not set both in the same combination):
# "" = no alignment filter for that axis.
# ALIGNMENT_RATIO_THRESHOLD_LIST: "0.5" -> --alignment_ratio_threshold 0.5 (run dir _align0.5)
ALIGNMENT_RATIO_THRESHOLD_LIST=("")
# ALIGN_MIN_LIST: "0.15" -> --align_min 0.15 (keep ratio >= value; run dir _alignmin0.15)
ALIGN_MIN_LIST=("")

# SLURM settings (match augment_finetune_sweep.sh)
TIME="4:00:00"
MEM="64G"
GPU="pro_6000:1"

# ============== JOB SUBMISSION ==============
echo "Submitting on_policy_distillation jobs..."
echo "=================================="
echo "Logs directory: ${LOG_DIR}"

job_count=0

for TASK in "${TASKS[@]}"; do
  for TEACHER_LR in "${TEACHER_LRS[@]}"; do
    for TEACHER_STEPS in "${TEACHER_STEPS_LIST[@]}"; do
      for BC_LR in "${BC_LRS[@]}"; do
        for BC_STEPS in "${BC_STEPS_LIST[@]}"; do
          for MAX_ITERS in "${MAX_ITERS_LIST[@]}"; do
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
                          if [[ -n "${full_exp}" ]]; then
                            suffix="${suffix}_full"
                          fi

                          JOB_NAME="opdist_t${TASK}_tlr${TEACHER_LR}_ts${TEACHER_STEPS}_bc${BC_LR}_bs${BC_STEPS}_mi${MAX_ITERS}_s${SEED}${suffix}"

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

                          echo "Submitting: task=${TASK} teacher_lr=${TEACHER_LR} teacher_steps=${TEACHER_STEPS} bc_lr=${BC_LR} bc_steps=${BC_STEPS} max_iters=${MAX_ITERS} seed=${SEED} teval_ep=${TEACHER_EVAL_EP} rollout_ep=${ROLLOUT_EP} align=${ALIGN_THRESH:-none} align_min=${ALIGN_MIN:-none} full_exp=${full_exp:-none} ${single_episode} ${finetune_type} ${cumulative} ${save_video_opt}"

                          sbatch <<EOF
#!/bin/bash
#SBATCH --job-name="${JOB_NAME}"
#SBATCH --time="${TIME}"
#SBATCH --mem-per-cpu="${MEM}"
#SBATCH --gpus="${GPU}"
#SBATCH --output="${LOG_DIR}/opdist_%j.out"
#SBATCH --error="${LOG_DIR}/opdist_%j.err"

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"

python meta_libero/scripts/on_policy_distillation.py \
  --task "${TASK}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --teacher_lr "${TEACHER_LR}" \
  --teacher_steps "${TEACHER_STEPS}" \
  --bc_lr "${BC_LR}" \
  --bc_steps "${BC_STEPS}" \
  --max_iters "${MAX_ITERS}" \
  --teacher_eval_episodes "${TEACHER_EVAL_EP}" \
  --rollout_episodes "${ROLLOUT_EP}" \
  ${single_episode} \
  ${FINETUNE_FLAG} \
  ${ALIGN_FLAG} \
  ${ALIGN_MIN_FLAG} \
  ${cumulative} \
  ${save_video_opt} \
  ${full_exp}
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

echo "=================================="
echo "Submitted ${job_count} jobs total"
echo "Check status with: squeue -u \$USER"
