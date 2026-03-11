# meta_libero Setup and Dataset Download

Before running any `meta_libero` scripts, create and activate a Python virtual environment.

## 1) Create a virtual environment

From the repository root:

```bash
cd /cluster/home/anmari/meta_vlas
python3 -m venv .venv
```

## 2) Activate the virtual environment

```bash
cd /cluster/home/anmari/meta_vlas
source .venv/bin/activate
```

## 3) Install required package for dataset download

```bash
pip install -U huggingface_hub
```

## 4) Download and convert to LeRobot dataset (optional)

Run the following commands:

```bash
bash /cluster/home/anmari/meta_vlas/download_libero_to_scratch.sh
cd /cluster/home/anmari/meta_vlas
python examples/libero/convert_libero_90_data_to_lerobot.py --data-dir /cluster/home/anmari/.cache/huggingface/hub/datasets--Embodied-CoT--embodied_features_and_demos_libero --push-to-hub
```

The download step fetches:

- `Embodied-CoT/embodied_features_and_demos_libero`

into:

- `/cluster/home/anmari/.cache/huggingface/hub/datasets--Embodied-CoT--embodied_features_and_demos_libero`

## Recommended alternative (skip conversion)

Instead of running `convert_libero_90_data_to_lerobot.py` (for example, just to set your own `REPO_NAME`), you can directly use the already converted dataset on Hugging Face:

- Dataset: [`antoniomari/libero_90`](https://huggingface.co/datasets/antoniomari/libero_90)

Download it locally with:

```bash
huggingface-cli download antoniomari/libero_90 --repo-type dataset --local-dir /cluster/home/anmari/.cache/huggingface/lerobot/antoniomari/libero_90
```

## 5) Run jobs

All commands below assume:

- `cd /cluster/home/anmari/meta_vlas`
- `source .venv/bin/activate`

### TTT evaluation (`ttt_evaluation.py`)

- Submit a sweep of jobs:

```bash
bash /cluster/home/anmari/meta_vlas/meta_libero/jobs/send_jobs.sh
```

- Submit a single configured SLURM job:

```bash
sbatch /cluster/home/anmari/meta_vlas/meta_libero/jobs/ttt.sh
```

- Run a single TTT job through `jobs/ttt.sh`:

  1) Edit `/cluster/home/anmari/meta_vlas/meta_libero/jobs/ttt.sh` and set your run parameters, for example:
     - `TASK_SUITE_NAME="libero_90"`
     - `TASK_ID=0`
     - `NUM_TRIALS=50`
     - `LR=2.5e-4`
     - `TTT_FREQUENCY=20`
     - `TTT_NUM_STEPS=5`
     - `TTT_K=6`
     - `SEED=1`
  2) Set optional mode flags in the same file:
     - `USE_LORA="--use-lora"` or `USE_LORA=""`
     - `ACTION_EXPERT_ONLY="--action-expert-only"` or `ACTION_EXPERT_ONLY=""`
     - `NO_RESET_POLICY="--no-reset-policy"` or `NO_RESET_POLICY=""`
     - `SAVE_VIDEO="--save-video"` or `SAVE_VIDEO=""`
  3) Submit:

```bash
sbatch /cluster/home/anmari/meta_vlas/meta_libero/jobs/ttt.sh
```

- Run a single TTT command directly in an interactive job (no `sbatch`) with `meta_update=reset` (default):

```bash
python meta_libero/scripts/ttt_evaluation.py --task_suite_name libero_90 --task_id 0 --num_trials 50 --lr 2.5e-4 --ttt_frequency 20 --ttt_num_steps 5 --ttt_k 6 --num_neighbors_fetch 32 --meta_update reset --seed 1 --use-lora --libero-90-dataset --save-video
```

- Run TTT with `meta_update=continual_ttt` (continual adaptation, no per-episode reset):

```bash
python meta_libero/scripts/ttt_evaluation.py --task_suite_name libero_90 --task_id 0 --num_trials 50 --lr 2.5e-4 --ttt_frequency 20 --ttt_num_steps 5 --ttt_k 6 --num_neighbors_fetch 32 --meta_update continual_ttt --seed 1 --use-lora --libero-90-dataset --save-video
```

- Run TTT with `meta_update=tt_reptile` (model merging enabled via `--merging_eps`):

```bash
python meta_libero/scripts/ttt_evaluation.py --task_suite_name libero_90 --task_id 0 --num_trials 50 --lr 2.5e-4 --ttt_frequency 20 --ttt_num_steps 5 --ttt_k 6 --num_neighbors_fetch 32 --meta_update tt_reptile --merging_eps 0.75 --seed 1 --use-lora --libero-90-dataset --save-video
```

### Fine-tuning (`finetune_single_task.py`)

- Submit a sweep of jobs:

```bash
bash /cluster/home/anmari/meta_vlas/meta_libero/jobs/ft_single_jobs.sh
```

- Run a single fine-tuning job through `jobs/ft_single_jobs.sh`:

  1) Edit `/cluster/home/anmari/meta_vlas/meta_libero/jobs/ft_single_jobs.sh` and set a single value in each relevant array, for example:
     - `TASK_IDS=(0)`
     - `SEEDS=(1)`
     - `LEARNING_RATES=(2.5e-05)`
     - `BATCH_SIZES=(32)`
     - `TOTAL_STEPS=(500)`
     - `EVAL_INTERVALS=(100)`
     - `WARMUP_STEPS=(0)`
  2) (Optional) adjust SLURM resources in the same file:
     - `TIME="24:00:00"`
     - `MEM="64G"`
     - `GPU="a100_80gb:1"`
  3) Submit:

```bash
bash /cluster/home/anmari/meta_vlas/meta_libero/jobs/ft_single_jobs.sh
```

- Run directly (no SLURM):

```bash
python meta_libero/scripts/finetune_single_task.py --task_suite_name libero_90 --task_id 0 --num_trials 50 --lr 2.5e-4 --batch_size 32 --total_steps 500 --eval_interval 100 --warmup_steps 0 --seed 1 --use-lora --libero-90-dataset
```

Check jobs/logs:

```bash
squeue -u $USER
ls /cluster/home/anmari/meta_vlas/meta_libero/logs
```

### Interactive Evaluation Web UI (`eval_ui_server.py`)

A local web UI for interactive LIBERO-90 evaluation: select a task, load the environment, optionally edit the task instruction (prompt), and run a single-episode rollout with Play/Pause/Reset controls.

**Prerequisites:**

```bash
pip install fastapi uvicorn
```

**Run the server:**

```bash
cd /cluster/home/anmari/meta_vlas
python meta_libero/scripts/eval_ui_server.py --host 0.0.0.0 --port 8765
```

Open in any browser: **http://localhost:8765** (or `http://<machine-ip>:8765` for remote access).

**Usage:**

1. Select a task from the dropdown (all 90 LIBERO-90 tasks).
2. Optionally edit the task instruction (defaults to the task’s instruction).
3. Click **Load environment**.
4. Use **Play** for continuous rollout, **Pause** to stop, **Step** for single steps, **Reset** to restart the episode.

The UI reuses the same policy inference and environment stepping logic as `ttt_evaluation.py` (no TTT adaptation).

## 6) Results folder structure

### TTT results

Root structure:

- `meta_libero/results/{ttt|ttt_base_model}/{dataset_libero_10|dataset_libero_90}[_no_reset_policy]/{lora|action_only|...}`

Inside each settings folder:

- `results_summary.csv` (single aggregate CSV for that settings folder)
- `{task_suite_name}_task_{task_id}/{run_subfolder}/...` (videos, plots, config)

Example run subfolder:

- `lr2.50e-4_freq20_steps5_k6_seed1`

CSV columns (`results_summary.csv`):

- `lr, ttt_frequency, ttt_num_steps, batch_size, seed, task_suite_name, task_id, success_rate, num_trials, use_lora, action_expert_only`

### Fine-tuning results

Root structure:

- `meta_libero/results/full_finetuning_{task_suite_name}/lr_{lr}{_base}_b{batch_size}`

Inside each settings folder:

- `results_summary.csv` (single aggregate CSV for that settings folder)
- `experiment_config.yaml`
- `videos/`
- `{task_suite_name}_{task_id}_losses.pdf`
- `{task_suite_name}_{task_id}_accuracy.pdf`

CSV columns (`results_summary.csv`):

- `lr, ttt_frequency, ttt_num_steps, batch_size, seed, task_suite_name, task_id, success_rate, num_trials, use_lora, action_expert_only`

Note: for fine-tuning rows, `ttt_frequency` and `ttt_num_steps` are set to `0` (no TTT).

## 7) `meta_libero` folder structure

Recommended mental model:

- `meta_libero/scripts/`
  - entrypoints for experiments
  - `ttt_evaluation.py`: TTT evaluation loop
  - `finetune_single_task.py`: single-task fine-tuning + evaluation
  - `eval_tasks.py`: task evaluation helper entrypoint
  - `eval_ui_server.py`: interactive web UI server for LIBERO-90 evaluation
  - `eval_ui.html`: frontend for the evaluation UI
- `meta_libero/jobs/`
  - SLURM launchers
  - `send_jobs.sh`: TTT sweeps
  - `ft_single_jobs.sh`: fine-tuning sweeps
  - `ttt.sh`: single TTT SLURM job template
- `meta_libero/notebooks/`
  - exploratory and analysis notebooks
  - `train_libero.ipynb`
  - `access_libero_samples.ipynb`
- `meta_libero/results/`
  - experiment outputs
  - per-settings aggregate CSV (`results_summary.csv`)
  - per-run folders (plots, videos, configs)
- `meta_libero/logs/`
  - SLURM stdout/stderr (`*.out`, `*.err`)
- `meta_libero/libero_dataset.py`
  - dataset wrapper + task filtering utilities
  - override hook for custom task-filtered data loaders
- `meta_libero/configs.py`
  - experiment config dataclasses and CLI override plumbing
- `meta_libero/utils.py`
  - shared model loading, policy creation, and eval/train helpers
- `meta_libero/nn_fetcher.py`
  - nearest-neighbor retrieval logic used by TTT
- `meta_libero/build_unified_faiss_index.py`
  - FAISS index build script for retrieval-based workflows
- `meta_libero/rendering.py`, `meta_libero/libero_ood.py`, `meta_libero/fix_checkpoint.py`
  - utilities for rendering, OOD analysis, and checkpoint maintenance

## 8) Export/setup improvements for another machine

To make this subproject easier to move and run elsewhere, prioritize:

- Replace hard-coded absolute paths with env vars:
  - `META_VLAS_ROOT`, `HF_HOME`, `HF_LEROBOT_HOME`, `RESULTS_DIR`, `LOG_DIR`
- Add one environment bootstrap command in repo root:
  - install Python deps and verify imports (`huggingface_hub`, `tensorflow_datasets`, `lerobot`, etc.)
- Add a single source-of-truth config file for cluster/local differences:
  - e.g., YAML or `.env` consumed by job scripts and Python entrypoints
- Standardize SLURM scripts to read shared defaults:
  - GPU type, memory, time, log path in one place
- Add a preflight command for portability checks:
  - HF auth, dataset visibility, writable cache/result directories, FAISS index presence
- Keep generated artifacts out of source export:
  - exclude `results/`, `logs/`, and large caches from transfer/packaging

Suggested quick export checklist:

1. Create and activate venv.
2. Install dependencies.
3. Set env vars for paths.
4. Run dataset download/convert.
5. Build FAISS index if needed.
6. Run one local smoke test command (non-SLURM).
7. Submit one SLURM test job.

### Environment variables used by scripts/jobs

You can override paths without editing code:

- `PROJECT_ROOT`: repo root path (used by SLURM job scripts)
- `META_VENV_PATH`: virtualenv path (default: `${PROJECT_ROOT}/.venv`)
- `META_LIBERO_LOG_DIR`: log directory (default: `${PROJECT_ROOT}/meta_libero/logs`)
- `META_LIBERO_RESULTS_DIR`: results root (default: `meta_libero/results`)
- `OPENPI_CHECKPOINT_DIR`: checkpoint directory (default: `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero`)
- `HF_HOME`: Hugging Face cache directory (default: `~/.cache/huggingface`)
- `HF_LEROBOT_HOME`: LeRobot cache directory (default: `${HF_HOME}/lerobot`)
- `LIBERO_DATASET_DIR`: local raw LIBERO dataset dir (used by `libero_dataset.py`)

Example:

```bash
export PROJECT_ROOT=/path/to/meta_vlas
export META_VENV_PATH=$PROJECT_ROOT/.venv
export META_LIBERO_LOG_DIR=$PROJECT_ROOT/meta_libero/logs
export META_LIBERO_RESULTS_DIR=$PROJECT_ROOT/meta_libero/results
export OPENPI_CHECKPOINT_DIR=$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero
export HF_HOME=$HOME/.cache/huggingface
export HF_LEROBOT_HOME=$HF_HOME/lerobot
export LIBERO_DATASET_DIR=/path/to/libero_datasets
```
