#!/usr/bin/env python3
"""
Build Unified FAISS Index for All Tasks

This script builds a SINGLE FAISS index containing data from all tasks in a dataset.
Unlike build_all_task_indices.py which creates separate indices per task,
this creates one unified index for cross-task retrieval.

Usage:
    python build_unified_faiss_index.py --dataset-repo lerobot/libero_10 --modalities image1 image2 text
"""

import argparse
import pickle
import gc
import functools
import jax
import jax.numpy as jnp
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import warnings
import dataclasses

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import faiss
from openpi.shared import image_tools
from openpi.models.model import IMAGE_RESOLUTION
from openpi.training import config as _config
from openpi.models import model as _model
from openpi.models.tokenizer import PaligemmaTokenizer
import openpi.shared.download as download
from openpi.training import data_loader as _data_loader


def clear_jax_cache():
    """Clear JAX memory cache to free GPU memory."""
    jax.clear_caches()
    gc.collect()


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
        print("Checkpoint is already correct (or has fewer samples than expected).")
        return False

    print(f"\nTruncating checkpoint from {current_samples} to {expected_samples} samples...")

    # Truncate embeddings and metadata
    all_embeddings = all_embeddings[:expected_samples]
    metadata = metadata[:expected_samples]

    # Recalculate processed_batches based on truncated data
    # This is approximate - we calculate based on samples
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


def load_model(config):
    """Load the pi0.5 model and return the image encoder, text encoder, and tokenizer."""
    print("Loading pi0.5 model...")
    checkpoint_gs_path = "gs://openpi-assets/checkpoints/pi05_libero"
    checkpoint_dir = download.maybe_download(checkpoint_gs_path)

    model = config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    return model


def create_jit_embedding_functions(image_encoder, text_encoder):
    """
    Create JIT-compiled embedding functions with encoders bound via partial.

    Args:
        image_encoder: The SigLIP image encoder
        text_encoder: The LLM text encoder

    Returns:
        Tuple of (image_embedding_fn, text_embedding_fn) that are JIT-compiled
    """
    # Define the core functions that will be JIT-compiled
    def _extract_image_embedding_core(images, encoder):
        image_tokens, _ = encoder(images, train=False)
        embeddings = jnp.mean(image_tokens, axis=1)  # Global average pooling
        return embeddings

    def _extract_text_embedding_core(tokenized_text, encoder):
        text_tokens = encoder(tokenized_text, method="embed")
        embedding = jnp.mean(text_tokens, axis=1)  # Pool across sequence
        return embedding

    # Create partial functions with encoders bound
    image_embedding_partial = functools.partial(_extract_image_embedding_core, encoder=image_encoder)
    text_embedding_partial = functools.partial(_extract_text_embedding_core, encoder=text_encoder)

    # JIT compile the partial functions
    image_embedding_jit = jax.jit(image_embedding_partial)
    text_embedding_jit = jax.jit(text_embedding_partial)

    print("✓ Created JIT-compiled embedding functions")

    return image_embedding_jit, text_embedding_jit




def extract_image_embedding(images, image_embedding_fn):
    """
    Extract image embeddings from pi0.5's SigLIP encoder.

    Args:
        images: Images array, shape (batch_size, h, w, c) or (h, w, c)
        image_embedding_fn: JIT-compiled embedding function (from create_jit_embedding_functions)

    Returns:
        Embedding vector(s) as numpy array: (batch_size, embed_dim) or (embed_dim,)
    """
    single_image = False
    if images.ndim == 3:
        single_image = True
        images = images[None, ...]

    # Call JIT-compiled function
    embeddings = image_embedding_fn(images)
    embeddings_np = np.array(embeddings)

    if single_image:
        embeddings_np = embeddings_np[0]

    return embeddings_np


def extract_text_embedding(tokenized_text, text_embedding_fn):
    """
    Extract text embedding from pi0.5's language model.

    Args:
        tokenized_text: Already tokenized text tensor (batch_size, seq_len)
        text_embedding_fn: JIT-compiled embedding function (from create_jit_embedding_functions)

    Returns:
        Embedding vector(s) as numpy array: (batch_size, embed_dim)
    """
    # Call JIT-compiled function
    embedding = text_embedding_fn(tokenized_text)
    embedding_np = np.array(embedding)

    return embedding_np


def build_unified_index(
    cache_dir,
    batch_size=256,
    modalities=["image1", "image2", "text"],
    resume=False,
    rebuild_only=False,
):
    """
    Build a unified FAISS index from all tasks in a dataset.

    Args:
        cache_dir: Directory to save the index
        batch_size: Batch size for processing
        modalities: List of modalities to include
        resume: Resume from checkpoint if available
        rebuild_only: If True, only rebuild index from checkpoint, skip all processing
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine dataset name from repo
    modality_str = "_".join(sorted(modalities))
    index_path = cache_dir / f"libero_unified_faiss_index_{modality_str}.index"
    metadata_path = cache_dir / f"libero_unified_faiss_metadata_{modality_str}.pkl"
    checkpoint_path = cache_dir / f"libero_unified_checkpoint_{modality_str}.pkl"

    print("="*70)
    print(f"Building Unified FAISS Index for libero")
    print("="*70)
    print(f"Modalities: {modalities}")
    print(f"Cache directory: {cache_dir}")
    print(f"Batch size: {batch_size}")
    if rebuild_only:
        print("Mode: REBUILD ONLY (from checkpoint)")
    print("="*70)

    # For rebuild_only, we don't need the model
    if not rebuild_only:
        # Load model
        config = _config.get_config("pi05_libero")
        model = load_model(config)
        image_encoder = model.PaliGemma.img
        text_encoder = model.PaliGemma.llm
        print("Model loaded successfully!")

        # Create JIT-compiled embedding functions
        image_embedding_fn, text_embedding_fn = create_jit_embedding_functions(image_encoder, text_encoder)
    else:
        # For rebuild_only, we still need config to get dataset size
        config = _config.get_config("pi05_libero")
        image_embedding_fn = None
        text_embedding_fn = None


    # For rebuild_only, skip data loader creation
    if not rebuild_only:
        # Override the dataset repo in config
        config = dataclasses.replace(
            config,
            batch_size=batch_size,  # Process one sample at a time for embedding extraction
        )

        # First, create a temporary data loader to get the dataset size
        temp_data_loader = _data_loader.create_data_loader(
            config,
            sharding=None,
            shuffle=False,
            num_batches=1,  # Just to get the dataset
        )
        # Get the underlying dataset
        dataset = temp_data_loader._data_loader._data_loader.dataset
        total_dataset_samples = len(dataset)
        print(f"Total samples in dataset: {total_dataset_samples}")

        # Calculate expected number of batches (accounting for drop_last=True in TorchDataLoader)
        # TorchDataLoader uses drop_last=True, so we need to calculate batches accordingly
        expected_batches = total_dataset_samples // batch_size  # Integer division (drop_last=True)
        expected_samples = expected_batches * batch_size
        print(f"Expected number of batches: {expected_batches} (batch_size={batch_size}, drop_last=True)")
        print(f"Expected number of samples: {expected_samples}")

        # Fix checkpoint if it has too many samples (from previous buggy run)
        if resume and checkpoint_path.exists():
            print(f"\nChecking checkpoint for truncation...")
            fix_checkpoint(checkpoint_path, total_dataset_samples, batch_size)

        # Create data loader with num_batches to prevent infinite repetition
        data_loader = _data_loader.create_data_loader(
            config,
            sharding=None,
            shuffle=False,  # Don't shuffle for consistent indexing
            num_batches=expected_batches,  # This prevents the dataloader from repeating!
        )
        print("Data loader created successfully!")
    else:
        # For rebuild_only, we don't need these
        data_loader = None
        expected_batches = 0
        expected_samples = 0
        total_dataset_samples = 0

    # Load checkpoint if resuming
    all_embeddings = []
    metadata = []
    processed_batches = 0
    initial_sample_count = 0

    # Check if we need to process more batches or can skip directly to index creation
    all_embeddings = []
    metadata = []
    processed_batches = 0
    initial_sample_count = 0
    skip_processing = False

    # Initialize dimensions (will be set during processing or from checkpoint)
    img1_dim = 0
    img2_dim = 0
    text_dim = 0
    total_dim = 0

    # For rebuild_only, load embeddings from checkpoint or existing index
    if rebuild_only:
        # Try checkpoint first, then fall back to existing index
        if checkpoint_path.exists():
            print(f"\nRebuild-only mode: Loading embeddings from checkpoint...")
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
                all_embeddings = checkpoint.get("embeddings", [])
                metadata = checkpoint.get("metadata", [])
                processed_batches = checkpoint.get("processed_batches", 0)

            print(f"Loaded {len(all_embeddings)} embeddings from checkpoint")
        elif index_path.exists():
            print(f"\nRebuild-only mode: Checkpoint not found, loading embeddings from existing index...")
            print(f"Loading index from {index_path}...")
            existing_index = faiss.read_index(str(index_path))
            num_vectors = existing_index.ntotal
            embedding_dim = existing_index.d

            print(f"Found {num_vectors} vectors with dimension {embedding_dim}")

            # Extract all embeddings from the index
            print("Extracting embeddings from index...")
            all_embeddings = []
            # Reconstruct embeddings one by one (FAISS reconstruct method)
            for i in tqdm(range(num_vectors), desc="Reconstructing embeddings"):
                emb = existing_index.reconstruct(i)
                all_embeddings.append(emb)

            print(f"Extracted {len(all_embeddings)} embeddings from index")

            # Set total_dim from index
            total_dim = embedding_dim
            print(f"Total embedding dimension from index: {total_dim}")

            # Initialize empty metadata - will be regenerated
            metadata = []
            processed_batches = 0
        else:
            raise FileNotFoundError(
                f"Neither checkpoint nor index found.\n"
                f"  Checkpoint: {checkpoint_path}\n"
                f"  Index: {index_path}\n"
                "Cannot rebuild without checkpoint or existing index. Run without --rebuild-only to create embeddings first."
            )

        # Try to get dimensions from existing metadata file if available
        if metadata_path.exists():
            print(f"Loading dimensions from existing metadata...")
            with open(metadata_path, "rb") as f:
                existing_meta = pickle.load(f)
                embedding_dims = existing_meta.get("embedding_dims", {})
                img1_dim = embedding_dims.get("image1", 0)
                img2_dim = embedding_dims.get("image2", 0)
                text_dim = embedding_dims.get("text", 0)
                total_dim_meta = embedding_dims.get("total", 0)
                # Only use total_dim from metadata if it's not already set from index
                if total_dim == 0:
                    total_dim = total_dim_meta
                print(f"Dimensions from metadata: img1={img1_dim}, img2={img2_dim}, text={text_dim}, total={total_dim}")

        # If total_dim not set, extract from embeddings
        if total_dim == 0 and len(all_embeddings) > 0:
            first_emb = np.array(all_embeddings[0])
            total_dim = int(first_emb.shape[0])
            print(f"Total embedding dimension from embeddings: {total_dim}")

        # If individual dimensions are missing (0) but we have total_dim, infer them
        if total_dim > 0 and (img1_dim == 0 or img2_dim == 0 or text_dim == 0):
            print(f"Inferring individual modality dimensions from total={total_dim}...")
            assert len(modalities) == 3 and "image1" in modalities and "image2" in modalities and "text" in modalities
            img1_dim = 2048
            img2_dim = 2048
            text_dim = total_dim - img1_dim - img2_dim
            print(f"Inferred dimensions: img1={img1_dim}, img2={img2_dim}, text={text_dim}")

        # Regenerate metadata if needed (simple indices)
        if len(metadata) != len(all_embeddings) or len(metadata) == 0:
            print(f"Regenerating metadata: {len(metadata)} -> {len(all_embeddings)}")
            metadata = []
            for i in range(len(all_embeddings)):
                metadata.append({
                    "sample_idx": i,
                    "batch_idx": i // batch_size,
                })

        batch_count = processed_batches
        sample_count = len(all_embeddings)
        skip_processing = True

        print(f"✓ Ready to rebuild index with {len(all_embeddings)} embeddings")

    elif resume and checkpoint_path.exists():
        print(f"\nResuming from checkpoint...")
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
            all_embeddings = checkpoint.get("embeddings", [])
            metadata = checkpoint.get("metadata", [])
            processed_batches = checkpoint.get("processed_batches", 0)
            initial_sample_count = len(all_embeddings)
        print(f"Loaded {initial_sample_count} existing embeddings from {processed_batches} batches")

        # Verify checkpoint is correct
        if initial_sample_count > expected_samples:
            print(f"WARNING: Checkpoint still has {initial_sample_count} samples, expected {expected_samples}")
            print("Truncating to expected number...")
            all_embeddings = all_embeddings[:expected_samples]
            metadata = metadata[:expected_samples]
            processed_batches = expected_batches
            initial_sample_count = len(all_embeddings)
            print(f"Truncated to {initial_sample_count} samples")

        # Check if we already have all the embeddings
        if processed_batches >= expected_batches and initial_sample_count >= expected_samples:
            print(f"\n✓ Checkpoint already contains all {initial_sample_count} embeddings!")
            print("Skipping dataloader iteration and proceeding directly to index creation...")
            skip_processing = True
            batch_count = processed_batches
            sample_count = initial_sample_count

            # Extract embedding dimensions from existing embeddings
            if len(all_embeddings) > 0:
                # Get dimensions from first embedding
                first_emb = np.array(all_embeddings[0])
                total_dim = int(first_emb.shape[0])
                print(f"Embedding dimension from checkpoint: {total_dim}")

                # Try to infer individual modality dimensions (approximate)
                # This is a heuristic - we can't perfectly determine without processing
                assert "image1" in modalities and "image2" in modalities and "text" in modalities
                img1_dim = 2048
                img2_dim = 2048
                text_dim = total_dim - img1_dim - img2_dim  # Remaining for text
        else:
            print(f"\nNeed to process more batches: {processed_batches}/{expected_batches}")

    # Initialize batch_count and sample_count
    batch_count = processed_batches
    sample_count = initial_sample_count

    # Process batches from dataloader only if needed
    if not skip_processing:
        print(f"\nProcessing dataset in batches of {batch_size}...")
        print("Iterating through dataloader...")
        print(f"Will process {expected_batches} batches ({expected_batches * batch_size} samples, may be less than {total_dataset_samples} due to drop_last=True)")

        # Track dimensions (will be determined from first batch)
        # Only reset if we're actually processing (not skipping)
        img1_dim = 0
        img2_dim = 0
        text_dim = 0
        total_dim = 0

        # Iterate through dataloader
        for batch in tqdm(data_loader, desc="Processing batches", total=expected_batches, initial=processed_batches):
            batch_count += 1

            # Skip if resuming and already processed
            if resume and batch_count <= processed_batches:
                continue

            # Unpack the batch tuple: (Observation, actions)
            observation, actions = batch

            current_batch_size = None

            # Extract embeddings for each modality
            embeddings_list = []

            # Image1 (agentview/base camera) embeddings
            if "image1" in modalities:
                # Observation.images is a dict with key "base_0_rgb"
                if "base_0_rgb" in observation.images:
                    img1_batch = observation.images["base_0_rgb"]  # (batch_size, h, w, c)
                    if current_batch_size is None:
                        current_batch_size = img1_batch.shape[0]

                    # Extract embeddings for the batch
                    img1_emb = extract_image_embedding(img1_batch, image_embedding_fn)  # (batch_size, embed_dim)
                    embeddings_list.append(img1_emb)

                    if img1_dim == 0:
                        img1_dim = img1_emb.shape[-1]
                        print(f"Image1 (base_0_rgb) embedding dimension: {img1_dim}")

            # Image2 (wrist camera) embeddings
            if "image2" in modalities:
                # Observation.images is a dict with key "left_wrist_0_rgb"
                if "left_wrist_0_rgb" in observation.images:
                    img2_batch = observation.images["left_wrist_0_rgb"]  # (batch_size, h, w, c)
                    if current_batch_size is None:
                        current_batch_size = img2_batch.shape[0]

                    # Extract embeddings for the batch
                    img2_emb = extract_image_embedding(img2_batch, image_embedding_fn)  # (batch_size, embed_dim)
                    embeddings_list.append(img2_emb)

                    if img2_dim == 0:
                        img2_dim = img2_emb.shape[-1]
                        print(f"Image2 (left_wrist_0_rgb) embedding dimension: {img2_dim}")

            # Text embeddings (from tokenized_prompt)
            if "text" in modalities:
                # Observation.tokenized_prompt contains the tokenized text
                if observation.tokenized_prompt is not None and observation.tokenized_prompt_mask is not None:
                    if current_batch_size is None:
                        current_batch_size = observation.tokenized_prompt.shape[0]

                    # Use the tokenized prompt directly with the text encoder
                    # The text encoder expects tokenized input
                    text_emb_batch = extract_text_embedding(
                        observation.tokenized_prompt,  # Already tokenized: (batch_size, seq_len)
                        text_embedding_fn  # JIT-compiled function
                    )  # (batch_size, embed_dim)

                    embeddings_list.append(text_emb_batch)

                    if text_dim == 0:
                        text_dim = text_emb_batch.shape[-1]
                        print(f"Text embedding dimension: {text_dim}")

            # Concatenate all modality embeddings
            if embeddings_list:
                # Concatenate along feature dimension
                combined_embeddings = np.concatenate(embeddings_list, axis=-1)  # (batch_size, total_embed_dim)

                if total_dim == 0:
                    total_dim = combined_embeddings.shape[-1]
                    print(f"Total embedding dimension: {total_dim}")

                # Get batch size from combined embeddings if not set
                if current_batch_size is None:
                    current_batch_size = combined_embeddings.shape[0]

                # Add each sample's embedding to the list
                for i in range(current_batch_size):
                    all_embeddings.append(combined_embeddings[i])

                    # Store metadata
                    meta_dict = {
                        "sample_idx": sample_count + i,
                        "batch_idx": batch_count - 1,
                    }

                    # Note: Observation doesn't directly contain task_index or text prompt
                    # These would need to come from the original dataset if needed
                    # For now, we just store the sample index

                    metadata.append(meta_dict)

                sample_count += current_batch_size

            # Clear cache periodically
            if batch_count % 10 == 0:
                clear_jax_cache()

            # Save checkpoint periodically
            if batch_count % 50 == 0:
                print(f"\nSaving checkpoint at batch {batch_count} ({len(all_embeddings)} embeddings, {sample_count}/{total_dataset_samples} samples)...")
                with open(checkpoint_path, "wb") as f:
                    pickle.dump({
                        "embeddings": all_embeddings,
                        "metadata": metadata,
                        "processed_batches": batch_count,
                    }, f)


    print(f"\nTotal embeddings extracted: {len(all_embeddings)}")
    print(f"Total samples processed: {sample_count}")
    print(f"Expected samples: {expected_batches * batch_size} (may be less due to drop_last=True)")

    # Note: sample_count may be slightly less than total_dataset_samples due to drop_last=True
    # This is expected behavior

    # Clear GPU cache
    print("Clearing GPU cache...")
    clear_jax_cache()

    # Convert to numpy array
    print("Converting embeddings to numpy array...")
    embeddings_array = np.array(all_embeddings).astype(np.float32)
    print(f"Embeddings array shape: {embeddings_array.shape}")

    # Normalize for cosine similarity
    print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings_array)

    # Create FAISS index
    print("Creating FAISS index...")
    embedding_dim = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_array)

    print(f"Index size: {index.ntotal}")

    # Save index and metadata
    print(f"\nSaving FAISS index to {index_path}...")
    faiss.write_index(index, str(index_path))

    print(f"Saving metadata to {metadata_path}...")

    embedding_dims_dict = {
        "image1": img1_dim,
        "image2": img2_dim,
        "text": text_dim,
        "total": total_dim,
    }

    with open(metadata_path, "wb") as f:
        pickle.dump({
            "metadata": metadata,
            "modalities": modalities,
            "embedding_dims": embedding_dims_dict,
            "total_samples": len(all_embeddings),
            "num_batches": batch_count,
        }, f)

    print(f"\nEmbedding dimensions metadata:")
    print(f"  image1: {img1_dim}")
    print(f"  image2: {img2_dim}")
    print(f"  text: {text_dim}")
    print(f"  total: {total_dim}")

    # Remove checkpoint (unless in rebuild-only mode, where we keep it for future rebuilds)
    if not rebuild_only and checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint removed")
    elif rebuild_only:
        print("Checkpoint preserved (rebuild-only mode)")

    print("\n" + "="*70)
    print("Unified FAISS index created successfully!")
    print("="*70)
    print(f"Total samples indexed: {len(all_embeddings)}")
    print(f"Number of batches processed: {batch_count}")
    print(f"Modalities used: {modalities}")
    print(f"Index file: {index_path}")
    print(f"Metadata file: {metadata_path}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Build unified FAISS index for all tasks in a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build unified index for libero_10 (all tasks)
  python build_unified_faiss_index.py

  # Build with only image modalities
  python build_unified_faiss_index.py  --modalities image1 image2

  # Resume from checkpoint
  python build_unified_faiss_index.py  --resume

  # Rebuild index/metadata from existing checkpoint or index (no reprocessing)
  python build_unified_faiss_index.py  --rebuild-only
"""
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for saving index (default: ~/.cache/libero_unified_faiss)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for processing images (default: 128)"
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["image1", "image2", "text"],
        default=["image1", "image2", "text"],
        help="Modalities to include in embeddings (default: all)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="Rebuild index and metadata from existing checkpoint or index file (skip all processing)"
    )

    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = str(Path.home() / ".cache" / "libero_unified_faiss")

    build_unified_index(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        modalities=args.modalities,
        resume=args.resume,
        rebuild_only=args.rebuild_only,
    )


if __name__ == "__main__":
    main()
