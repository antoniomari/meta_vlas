#!/bin/bash
# Download Embodied-CoT LIBERO dataset with Hugging Face API.

set -euo pipefail

REPO_ID="Embodied-CoT/embodied_features_and_demos_libero"
DOWNLOAD_DIR="/cluster/home/anmari/.cache/huggingface/hub/datasets--Embodied-CoT--embodied_features_and_demos_libero"

cd /cluster/home/anmari/meta_vlas
mkdir -p "$DOWNLOAD_DIR"

python - <<'PY'
from huggingface_hub import snapshot_download

repo_id = "Embodied-CoT/embodied_features_and_demos_libero"
local_dir = "/cluster/home/anmari/.cache/huggingface/hub/datasets--Embodied-CoT--embodied_features_and_demos_libero"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY

echo "Download complete from ${REPO_ID}"
echo "Local path: ${DOWNLOAD_DIR}"

