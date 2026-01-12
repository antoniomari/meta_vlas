#!/bin/bash
# Script to download LIBERO datasets to scratch space

# Set the download directory to scratch
DOWNLOAD_DIR="/cluster/scratch/anmari/libero_datasets"

# Change to the meta_vlas directory
cd /cluster/home/anmari/meta_vlas

# Make sure the download directory exists
mkdir -p "$DOWNLOAD_DIR"

# Download all datasets (or specify a specific one: libero_goal, libero_spatial, libero_object, libero_100)
# For example, to download just one: --datasets libero_10
python third_party/libero/benchmark_scripts/download_libero_datasets.py \
    --download-dir "$DOWNLOAD_DIR" \
    --datasets all

echo "Download complete! Datasets are in: $DOWNLOAD_DIR"

