#!/usr/bin/env python3
"""
Fix checkpoint that has too many samples by truncating to the correct number.

This script is useful if you ran build_unified_faiss_index.py before the fix
and it processed too many samples (e.g., 1.2M instead of 200k).

Usage:
    python fix_checkpoint.py --cache-dir ~/.cache/libero_unified_faiss --batch-size 128
"""

import argparse
import pickle
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import dataclasses


def get_dataset_size(config, batch_size):
    """Get the total number of samples in the dataset."""
    config = dataclasses.replace(config, batch_size=batch_size)
    temp_data_loader = _data_loader.create_data_loader(
        config,
        sharding=None,
        shuffle=False,
        num_batches=1,  # Just to get the dataset
    )
    dataset = temp_data_loader._data_loader._data_loader.dataset
    return len(dataset)


def fix_checkpoint(checkpoint_path, total_dataset_samples, batch_size):
    """
    Fix a checkpoint that has too many samples by truncating to the correct number.

    Args:
        checkpoint_path: Path to the checkpoint file
        total_dataset_samples: Total number of samples in the dataset
        batch_size: Batch size used for processing

    Returns:
        True if checkpoint was fixed, False if it was already correct or didn't exist
    """
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return False

    print(f"\nChecking checkpoint: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    all_embeddings = checkpoint.get("embeddings", [])
    metadata = checkpoint.get("metadata", [])
    processed_batches = checkpoint.get("processed_batches", 0)

    current_samples = len(all_embeddings)
    expected_batches = total_dataset_samples // batch_size
    expected_samples = expected_batches * batch_size

    print(f"Current checkpoint: {current_samples} samples, {processed_batches} batches")
    print(f"Expected: {expected_samples} samples, {expected_batches} batches")

    if current_samples <= expected_samples:
        print("✓ Checkpoint is already correct (or has fewer samples than expected).")
        return False

    print(f"\nTruncating checkpoint from {current_samples} to {expected_samples} samples...")
    print(f"This will remove {current_samples - expected_samples} excess samples.")

    # Truncate embeddings and metadata
    all_embeddings = all_embeddings[:expected_samples]
    metadata = metadata[:expected_samples]

    # Recalculate processed_batches based on truncated data
    processed_batches = expected_batches

    # Save fixed checkpoint
    with open(checkpoint_path, "wb") as f:
        pickle.dump({
            "embeddings": all_embeddings,
            "metadata": metadata,
            "processed_batches": processed_batches,
        }, f)

    print(f"✓ Checkpoint fixed: {len(all_embeddings)} samples, {processed_batches} batches")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix checkpoint that has too many samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory containing checkpoint (default: ~/.cache/libero_unified_faiss)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used for processing (default: 128)"
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["image1", "image2", "text"],
        default=["image1", "image2", "text"],
        help="Modalities used (default: all)"
    )

    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = str(Path.home() / ".cache" / "libero_unified_faiss")

    cache_dir = Path(args.cache_dir)
    modality_str = "_".join(sorted(args.modalities))
    checkpoint_path = cache_dir / f"libero_unified_checkpoint_{modality_str}.pkl"

    print("="*70)
    print("Fix Checkpoint Script")
    print("="*70)
    print(f"Cache directory: {cache_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Modalities: {args.modalities}")
    print("="*70)

    # Get dataset size
    print("\nLoading config and getting dataset size...")
    config = _config.get_config("pi05_libero")
    total_dataset_samples = get_dataset_size(config, args.batch_size)
    print(f"Total samples in dataset: {total_dataset_samples}")

    # Fix checkpoint
    fixed = fix_checkpoint(checkpoint_path, total_dataset_samples, args.batch_size)

    if fixed:
        print("\n" + "="*70)
        print("Checkpoint fixed successfully!")
        print("You can now run build_unified_faiss_index.py with --resume to complete the index.")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("No fix needed or checkpoint not found.")
        print("="*70)


if __name__ == "__main__":
    main()
