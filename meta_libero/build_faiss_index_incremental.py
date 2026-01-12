#!/usr/bin/env python3
"""
Incremental FAISS Index Builder for LIBERO Dataset

This script builds a FAISS index from libero_90 dataset incrementally,
allowing for interruption and resumption. It saves progress after each file
and can resume from the last processed file.

Usage:
    python build_faiss_index_incremental.py [--resume] [--batch-size BATCH_SIZE]
"""

import argparse
import pickle
import gc
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm

import faiss

# Add project root to path (now one level deeper in libero subfolder)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from openpi.shared import image_tools
from openpi.models.model import IMAGE_RESOLUTION
from openpi.training import config as _config
from openpi.models import model as _model
from openpi.models.tokenizer import PaligemmaTokenizer
import openpi.shared.download as download
import json


def clear_jax_cache():
    """Clear JAX memory cache to free GPU memory."""
    jax.clear_caches()
    gc.collect()


def extract_image_embedding(images, image_encoder):
    """
    Extract image embeddings from pi0.5's SigLIP encoder.
    Can process a single image or a batch of images.

    Args:
        images: numpy array of shape (H, W, 3) or (N, H, W, 3) with values in [0, 255] (uint8 or float)
        image_encoder: The SigLIP image encoder from the model

    Returns:
        Embedding vector(s) as numpy array:
        - If single image: shape (embedding_dim,)
        - If batch: shape (N, embedding_dim)
    """
    # Handle single image case: add batch dimension
    single_image = False
    if images.ndim == 3:
        single_image = True
        images = images[None, ...]  # (1, H, W, 3)

    batch_size = images.shape[0]

    # Step 1: Ensure images are uint8 in [0, 255] range
    if images.dtype != np.uint8:
        if images.max() <= 1.0:
            # Images are in [0, 1] range, convert to [0, 255]
            images = (images * 255).astype(np.uint8)
        else:
            images = images.astype(np.uint8)

    # Step 2: Resize to model's expected size (224, 224) using resize_with_pad
    images_resized = []
    for i in range(batch_size):
        img_resized = image_tools.resize_with_pad(images[i], IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1])
        images_resized.append(img_resized)
    images_resized = np.array(images_resized)  # (N, 224, 224, 3)

    # Step 3: Convert to float32 and normalize to [-1, 1] range
    images_normalized = images_resized.astype(jnp.float32) / 255.0 * 2.0 - 1.0  # (N, 224, 224, 3)

    # Step 4: Get image tokens from encoder
    image_tokens, _ = image_encoder(images_normalized, train=False)

    # Step 5: Pool the tokens using global average pooling
    embeddings = jnp.mean(image_tokens, axis=1)  # Global average pooling: (batch, embedding_dim)

    # Step 6: Convert back to numpy
    embeddings_np = np.array(embeddings)  # Shape: (N, embedding_dim)

    # Step 7: Remove batch dimension if single image was provided
    if single_image:
        embeddings_np = embeddings_np[0]  # Shape: (embedding_dim,)

    return embeddings_np


def extract_text_embedding(text, tokenizer, text_encoder):
    """
    Extract text embedding from pi0.5's language model.

    Args:
        text: String text to embed
        tokenizer: PaligemmaTokenizer instance
        text_encoder: The LLM encoder from model.PaliGemma.llm

    Returns:
        Embedding vector as numpy array: shape (embedding_dim,)
    """
    # Tokenize the text
    tokens, mask = tokenizer.tokenize(text, state=None)
    tokens = jnp.array(tokens[None, ...])  # Add batch dimension: (1, seq_len)
    mask = jnp.array(mask[None, ...])  # (1, seq_len)

    # Get text embeddings from LLM
    text_tokens = text_encoder(tokens, method="embed")  # (1, seq_len, embedding_dim)

    # Pool using mean (mask out padding tokens)
    mask_expanded = mask[..., None]  # (1, seq_len, 1)
    masked_tokens = text_tokens * mask_expanded
    embedding = jnp.sum(masked_tokens, axis=1) / jnp.sum(mask_expanded, axis=1)  # (1, embedding_dim)

    # Convert to numpy and remove batch dimension
    embedding_np = np.array(embedding[0])  # (embedding_dim,)

    return embedding_np


def load_model():
    """Load the pi0.5 model and return the image encoder, text encoder, and tokenizer."""
    print("Loading pi0.5 model...")
    train_config = _config.get_config("pi05_libero")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")

    print("Loading model weights...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    image_encoder = model.PaliGemma.img
    text_encoder = model.PaliGemma.llm

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = PaligemmaTokenizer(max_len=48)

    print("Model loaded successfully!")
    return image_encoder, text_encoder, tokenizer


def get_processed_files(checkpoint_path):
    """Get set of already processed file names from checkpoint."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        return set(checkpoint.get("processed_files", []))
    return set()


def save_checkpoint(checkpoint_path, processed_files, all_embeddings, metadata, current_file_idx):
    """Save checkpoint with current progress."""
    checkpoint = {
        "processed_files": list(processed_files),
        "all_embeddings": all_embeddings,
        "metadata": metadata,
        "current_file_idx": current_file_idx,
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved: {len(processed_files)} files processed, {len(all_embeddings)} embeddings")


def load_checkpoint(checkpoint_path):
    """Load checkpoint and return embeddings, metadata, and processed files."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        print(f"Loaded checkpoint: {len(checkpoint['processed_files'])} files already processed")
        return (
            checkpoint.get("all_embeddings", []),
            checkpoint.get("metadata", []),
            set(checkpoint.get("processed_files", [])),
        )
    return [], [], set()


def build_index_incremental(
    dataset_dir,
    cache_dir,
    batch_size=64,
    resume=False,
    save_every_n_files=1,
    use_gpu=False,
    include_text=True,
    embedding_mode="concatenate",
):
    """
    Build FAISS index incrementally from libero_90 dataset.

    Args:
        dataset_dir: Path to libero_datasets directory
        cache_dir: Path to cache directory for saving index and checkpoints
        batch_size: Batch size for processing images
        resume: Whether to resume from checkpoint
        save_every_n_files: Save checkpoint after every N files
        use_gpu: Whether to use GPU for FAISS indexing
        include_text: Whether to include text embeddings from goal/prompt
        embedding_mode: How to combine image and text embeddings:
            - "concatenate": Concatenate image and text embeddings (default)
            - "image_only": Only use image embeddings
            - "text_only": Only use text embeddings
    """
    # Setup paths
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    index_path = cache_dir / "libero_90_faiss_index.index"
    metadata_path = cache_dir / "libero_90_faiss_metadata.pkl"
    checkpoint_path = cache_dir / "faiss_index_checkpoint.pkl"

    libero_90_path = Path(dataset_dir) / "libero_90"
    hdf5_files_90 = sorted(list(libero_90_path.glob("*.hdf5")))

    if not hdf5_files_90:
        print("ERROR: No libero_90 HDF5 files found!")
        return

    print(f"Found {len(hdf5_files_90)} HDF5 files in libero_90")

    # Load model
    image_encoder, text_encoder, tokenizer = load_model()

    # Get text embedding from first file to determine dimensions
    if include_text and embedding_mode != "image_only":
        with h5py.File(hdf5_files_90[0], "r") as f:
            problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
            language_instruction = "".join(problem_info.get("language_instruction", []))
            if language_instruction:
                test_text_emb = extract_text_embedding(language_instruction, tokenizer, text_encoder)
                text_embedding_dim = test_text_emb.shape[0]
                print(f"Text embedding dimension: {text_embedding_dim}")
            else:
                print("Warning: No language_instruction found, disabling text embeddings")
                include_text = False

    # Load checkpoint if resuming
    if resume and checkpoint_path.exists():
        all_embeddings, metadata, processed_files = load_checkpoint(checkpoint_path)
        print(f"Resuming: {len(processed_files)} files already processed, {len(all_embeddings)} embeddings")
    else:
        all_embeddings = []
        metadata = []
        processed_files = set()
        if checkpoint_path.exists():
            checkpoint_path.unlink()  # Remove old checkpoint if not resuming

    # Check GPU availability for FAISS
    gpu_available = False
    try:
        if hasattr(faiss, 'get_num_gpus'):
            num_gpus = faiss.get_num_gpus()
            gpu_available = num_gpus > 0 and use_gpu
            if gpu_available:
                print(f"FAISS GPU support: {num_gpus} GPU(s) available")
    except:
        pass

    # Process files
    image_batch = []
    metadata_batch = []
    total_frames = len(all_embeddings)

    # Get text embedding once per file (same goal for all frames in a file)
    file_text_embeddings = {}  # Cache text embeddings per file

    for file_idx, hdf5_file in enumerate(hdf5_files_90):
        # Skip if already processed
        if hdf5_file.name in processed_files:
            print(f"Skipping already processed file {file_idx+1}/{len(hdf5_files_90)}: {hdf5_file.name}")
            continue

        print(f"\nProcessing file {file_idx+1}/{len(hdf5_files_90)}: {hdf5_file.name}")

        # Extract text embedding for this file (once per file)
        text_embedding = None
        language_instruction = None
        if include_text and embedding_mode != "image_only":
            with h5py.File(hdf5_file, "r") as f:
                problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
                language_instruction = "".join(problem_info.get("language_instruction", []))
                if language_instruction:
                    text_embedding = extract_text_embedding(language_instruction, tokenizer, text_encoder)
                    file_text_embeddings[hdf5_file.name] = text_embedding
                    print(f"  Goal: {language_instruction[:80]}...")

        try:
            with h5py.File(hdf5_file, "r") as f:
                demos = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
                if demos:
                    inds = np.argsort([int(elem[5:]) for elem in demos])
                    demos = [demos[i] for i in inds]

                camera_view = "agentview_rgb"

                # Create progress bar for episodes
                episode_pbar = tqdm(
                    enumerate(demos),
                    total=len(demos),
                    desc=f"File {file_idx+1}/{len(hdf5_files_90)}",
                    unit="episode",
                    leave=False
                )

                for episode_idx, demo_key in episode_pbar:
                    demo_data = f["data"][demo_key]

                    # Get only agentview_rgb camera view
                    if "obs" in demo_data and camera_view in demo_data["obs"]:
                        images = np.array(demo_data["obs"][camera_view])
                        num_steps = images.shape[0]

                        for step_idx in range(num_steps):
                            image = images[step_idx]

                            # Ensure image is in [0, 255] range
                            if image.max() <= 1.0:
                                image = (image * 255).astype(np.uint8)
                            else:
                                image = image.astype(np.uint8)

                            # Add to batch
                            image_batch.append(image)
                            metadata_batch.append({
                                "file_idx": file_idx,
                                "file_name": hdf5_file.name,
                                "episode_idx": episode_idx,
                                "demo_key": demo_key,
                                "step_idx": step_idx,
                                "camera_view": camera_view,
                                "language_instruction": language_instruction
                            })

                            # Process batch when it reaches batch_size
                            if len(image_batch) >= batch_size:
                                    batch_images = np.array(image_batch)
                                    batch_img_embeddings = extract_image_embedding(batch_images, image_encoder)

                                    # Combine image and text embeddings
                                    if include_text and embedding_mode == "concatenate" and text_embedding is not None:
                                        # Concatenate image and text embeddings
                                        batch_combined = []
                                        for img_emb in batch_img_embeddings:
                                            combined = np.concatenate([img_emb, text_embedding])
                                            batch_combined.append(combined)
                                        batch_embeddings = np.array(batch_combined)
                                    elif include_text and embedding_mode == "text_only" and text_embedding is not None:
                                        # Use only text embeddings (repeat for each image in batch)
                                        batch_embeddings = np.tile(text_embedding, (len(batch_img_embeddings), 1))
                                    else:
                                        # Use only image embeddings
                                        batch_embeddings = batch_img_embeddings

                                    all_embeddings.extend(batch_embeddings)
                                    metadata.extend(metadata_batch)
                                    total_frames += len(image_batch)

                                    # Clear batch and free memory
                                    del batch_images, batch_img_embeddings, batch_embeddings
                                    image_batch = []
                                    metadata_batch = []

                                    # Update progress bar with frame count
                                    episode_pbar.set_postfix({"total_frames": total_frames, "episode": episode_idx+1})

                                    # Clear GPU cache periodically
                                    if total_frames % 200 == 0:
                                        clear_jax_cache()

                episode_pbar.close()

            # Mark file as processed
            processed_files.add(hdf5_file.name)

            # Process remaining images in batch
            if len(image_batch) > 0:
                batch_images = np.array(image_batch)
                batch_img_embeddings = extract_image_embedding(batch_images, image_encoder)

                # Combine image and text embeddings
                if include_text and embedding_mode == "concatenate" and text_embedding is not None:
                    batch_combined = []
                    for img_emb in batch_img_embeddings:
                        combined = np.concatenate([img_emb, text_embedding])
                        batch_combined.append(combined)
                    batch_embeddings = np.array(batch_combined)
                elif include_text and embedding_mode == "text_only" and text_embedding is not None:
                    batch_embeddings = np.tile(text_embedding, (len(batch_img_embeddings), 1))
                else:
                    batch_embeddings = batch_img_embeddings

                all_embeddings.extend(batch_embeddings)
                metadata.extend(metadata_batch)
                total_frames += len(image_batch)
                del batch_images, batch_img_embeddings, batch_embeddings
                image_batch = []
                metadata_batch = []

            # Save checkpoint periodically
            if (file_idx + 1) % save_every_n_files == 0:
                save_checkpoint(checkpoint_path, processed_files, all_embeddings, metadata, file_idx)
                clear_jax_cache()

            print(f"  Completed file {file_idx+1}/{len(hdf5_files_90)}: {total_frames} total frames processed")

        except Exception as e:
            print(f"ERROR processing file {hdf5_file.name}: {e}")
            print("Saving checkpoint before continuing...")
            save_checkpoint(checkpoint_path, processed_files, all_embeddings, metadata, file_idx)
            raise

    # Process final batch if any
    if len(image_batch) > 0:
        batch_images = np.array(image_batch)
        batch_img_embeddings = extract_image_embedding(batch_images, image_encoder)

        # Get text embedding for the last file if needed
        if include_text and embedding_mode != "image_only" and len(hdf5_files_90) > 0:
            last_file = hdf5_files_90[-1]
            if last_file.name in file_text_embeddings:
                text_emb = file_text_embeddings[last_file.name]
            else:
                with h5py.File(last_file, "r") as f:
                    problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
                    language_instruction = "".join(problem_info.get("language_instruction", []))
                    if language_instruction:
                        text_emb = extract_text_embedding(language_instruction, tokenizer, text_encoder)
                    else:
                        text_emb = None
        else:
            text_emb = None

        # Combine embeddings
        if include_text and embedding_mode == "concatenate" and text_emb is not None:
            batch_combined = []
            for img_emb in batch_img_embeddings:
                combined = np.concatenate([img_emb, text_emb])
                batch_combined.append(combined)
            batch_embeddings = np.array(batch_combined)
        elif include_text and embedding_mode == "text_only" and text_emb is not None:
            batch_embeddings = np.tile(text_emb, (len(batch_img_embeddings), 1))
        else:
            batch_embeddings = batch_img_embeddings

        all_embeddings.extend(batch_embeddings)
        metadata.extend(metadata_batch)
        total_frames += len(image_batch)
        del batch_images, batch_img_embeddings, batch_embeddings

    print(f"\nTotal frames processed: {total_frames}")
    print(f"Total embeddings: {len(all_embeddings)}")

    # Final GPU cleanup
    print("Clearing GPU cache...")
    clear_jax_cache()

    # Convert to numpy array
    embeddings_array = np.array(all_embeddings).astype(np.float32)
    print(f"Embeddings array shape: {embeddings_array.shape}")

    # Normalize embeddings for cosine similarity (L2 normalization)
    faiss.normalize_L2(embeddings_array)

    # Create FAISS index
    embedding_dim = embeddings_array.shape[1]
    index_cpu = faiss.IndexFlatIP(embedding_dim)

    # Use GPU if available
    if gpu_available:
        try:
            print("\nUsing GPU for indexing...")
            gpu_resource = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resource, 0, index_cpu)
            index.add(embeddings_array)
            print(f"Index size: {index.ntotal}")
            print("Converting GPU index to CPU for saving...")
            index_cpu = faiss.index_gpu_to_cpu(index)
        except Exception as e:
            print(f"Failed to use GPU: {e}. Using CPU.")
            index_cpu.add(embeddings_array)
    else:
        print("\nUsing CPU for indexing...")
        index_cpu.add(embeddings_array)

    print(f"Index size: {index_cpu.ntotal}")

    # Save index and metadata
    print(f"\nSaving FAISS index to {index_path}...")
    faiss.write_index(index_cpu, str(index_path))
    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    # Remove checkpoint file after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint file removed (index creation complete)")

    print("\nFAISS index created and saved successfully!")
    print(f"Index: {index_path}")
    print(f"Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index incrementally from libero_90 dataset")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/cluster/scratch/anmari/libero_datasets",
        help="Path to libero_datasets directory"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(Path.home() / ".cache" / "libero_faiss"),
        help="Path to cache directory for saving index"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for processing images (default: 16)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint after every N files (default: 1)"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for FAISS indexing (requires faiss-gpu)"
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        default=True,
        help="Include text embeddings from goal/prompt (default: True)"
    )
    parser.add_argument(
        "--no-text",
        dest="include_text",
        action="store_false",
        help="Disable text embeddings (image only)"
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        choices=["concatenate", "image_only", "text_only"],
        default="concatenate",
        help="How to combine image and text embeddings: concatenate (default), image_only, or text_only"
    )

    args = parser.parse_args()

    build_index_incremental(
        dataset_dir=args.dataset_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        resume=args.resume,
        save_every_n_files=args.save_every,
        use_gpu=args.use_gpu,
        include_text=args.include_text,
        embedding_mode=args.embedding_mode,
    )


if __name__ == "__main__":
    main()

