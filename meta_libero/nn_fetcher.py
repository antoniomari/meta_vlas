#!/usr/bin/env python3
"""
Nearest Neighbor Fetcher for FAISS Index

This module provides a class for fetching top-k nearest neighbors from a FAISS index
with support for multi-modal queries (image1, image2, text) and masking.
"""

import pickle
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import faiss

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openpi.training import config as _config
from openpi.models import model as _model
from openpi.shared import image_tools
import openpi.shared.download as download
import torch.utils.data as torch_data
# Import embedding function creation from build_unified_faiss_index to ensure consistency
from meta_libero.build_unified_faiss_index import create_jit_embedding_functions


class NearestNeighborFetcher:
    """
    Fetch top-k nearest neighbors from a FAISS index.

    Supports multi-modal queries with selective modality usage via masking.
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        model: _model.BaseModel,
    ):
        """
        Initialize the nearest neighbor fetcher.

        Args:
            index_path: Path to the FAISS index file
            metadata_path: Path to the metadata pickle file
            model: Pi0.5 model
        """
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        # Load index
        print(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(str(self.index_path))

        # Load metadata
        print(f"Loading metadata from {self.metadata_path}...")
        with open(self.metadata_path, "rb") as f:
            meta = pickle.load(f)
            self.metadata = meta["metadata"]
            self.modalities = meta["modalities"]
            self.embedding_dims = meta["embedding_dims"]
            self.total_samples = meta["total_samples"]

        print(f"Loaded index with {self.index.ntotal} samples")
        print(f"Modalities: {self.modalities}")
        print(f"Embedding dimensions: {self.embedding_dims}")
        # Extract encoders
        self.image_encoder = model.PaliGemma.img
        self.text_encoder = model.PaliGemma.llm

        # Create JIT-compiled embedding functions using the same functions as build_unified_faiss_index
        # This ensures consistency between index building and querying
        self.image_embedding_fn, self.text_embedding_fn = create_jit_embedding_functions(
            self.image_encoder,
            self.text_encoder
        )

        print("✓ NearestNeighborFetcher initialized successfully")

    def _extract_image_embedding(self, images: np.ndarray) -> np.ndarray:
        """
        Extract image embeddings.

        Args:
            images: Images array, shape (batch_size, h, w, c) or (h, w, c)

        Returns:
            Embeddings array, shape (batch_size, embed_dim) or (embed_dim,)
        """
        # Add batch dimension if single image
        if images.ndim == 3:
            images = images[None, ...]
            single_image = True
        else:
            single_image = False

        # Convert to JAX and call JIT function
        images_jax = jnp.asarray(images)
        embeddings_jax = self.image_embedding_fn(images_jax)
        embeddings = np.array(embeddings_jax)

        # Remove batch dimension if single image
        if single_image:
            embeddings = embeddings[0]

        return embeddings

    def _extract_text_embedding(self, tokenized_text) -> np.ndarray:
        """
        Extract text embeddings.

        Args:
            tokenized_text: Tokenized text array, shape (batch_size, seq_len) or (seq_len,)

        Returns:
            Embeddings array, shape (batch_size, embed_dim) or (embed_dim,)
        """
        # Add batch dimension if single text
        if tokenized_text.ndim == 1:
            tokenized_text = tokenized_text[None, ...]
            single_text = True
        else:
            single_text = False

        # Convert to JAX array
        if isinstance(tokenized_text, np.ndarray):
            tokenized_text_jax = jnp.asarray(tokenized_text.astype(np.int32))
        else:
            tokenized_text_jax = tokenized_text

        # Call JIT function directly
        embedding_jax = self.text_embedding_fn(tokenized_text_jax)
        embedding = np.array(embedding_jax)

        # Remove batch dimension if single text
        if single_text:
            embedding = embedding[0]

        return embedding

    def create_query_embedding(
        self,
        observation,
        use_modalities: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Create a query embedding from observation data.

        Args:
            observation: Observation object/dict with 'images' and 'tokenized_prompt'
            use_modalities: List of modalities to use for query. If None, uses all available.
                           Can be any combination of ["image1", "image2", "text"]

        Returns:
            Query embedding array with shape (total_embed_dim,)
            Parts corresponding to unused modalities are zeroed out.
        """
        if use_modalities is None:
            use_modalities = self.modalities

        # Validate modalities
        for mod in use_modalities:
            if mod not in self.modalities:
                raise ValueError(f"Modality '{mod}' not available in index. Available: {self.modalities}")

        # Extract data from observation
        image1 = None
        image2 = None
        tokenized_text = None

        if hasattr(observation, 'images'):
            # It's an Observation object
            if "base_0_rgb" in observation.images:
                img = np.array(observation.images["base_0_rgb"])
                # Check for batch dimension
                if img.ndim == 4:
                    if img.shape[0] > 1:
                        raise ValueError(f"Image1 has batch size {img.shape[0]}, expected single image (batch size 1 or no batch dimension)")
                    image1 = img[0]  # Take first element if batch dimension exists
                else:
                    image1 = img
            if "left_wrist_0_rgb" in observation.images:
                img = np.array(observation.images["left_wrist_0_rgb"])
                # Check for batch dimension
                if img.ndim == 4:
                    if img.shape[0] > 1:
                        raise ValueError(f"Image2 has batch size {img.shape[0]}, expected single image (batch size 1 or no batch dimension)")
                    image2 = img[0]  # Take first element if batch dimension exists
                else:
                    image2 = img
            if hasattr(observation, 'tokenized_prompt') and observation.tokenized_prompt is not None:
                txt = np.array(observation.tokenized_prompt)
                # Check for batch dimension
                if txt.ndim == 2:
                    if txt.shape[0] > 1:
                        raise ValueError(f"Tokenized text has batch size {txt.shape[0]}, expected single text (batch size 1 or no batch dimension)")
                    tokenized_text = txt[0]  # Take first element if batch dimension exists
                else:
                    tokenized_text = txt
        elif isinstance(observation, dict):
            # It's a dict (e.g., from policy)
            if "observation/image" in observation:
                img = np.array(observation["observation/image"])
                if img.ndim == 4:
                    if img.shape[0] > 1:
                        raise ValueError(f"Image1 has batch size {img.shape[0]}, expected single image (batch size 1 or no batch dimension)")
                    image1 = img[0]
                else:
                    image1 = img
            if "observation/wrist_image" in observation:
                img = np.array(observation["observation/wrist_image"])
                if img.ndim == 4:
                    if img.shape[0] > 1:
                        raise ValueError(f"Image2 has batch size {img.shape[0]}, expected single image (batch size 1 or no batch dimension)")
                    image2 = img[0]
                else:
                    image2 = img

        # Build embedding parts in order matching the index
        # Simple logic: for each modality, if it's in use_modalities and data is available, extract it; otherwise use zeros
        embeddings_list = []

        for modality in self.modalities:
            expected_dim = self.embedding_dims[modality]

            if modality == "image1":
                if modality in use_modalities and image1 is not None:
                    emb = self._extract_image_embedding(image1)
                else:
                    emb = np.zeros(expected_dim, dtype=np.float32)

                assert emb.shape[0] == expected_dim, f"Image1 embedding shape {emb.shape} does not match expected dimension {expected_dim}"

            elif modality == "image2":
                if modality in use_modalities and image2 is not None:
                    emb = self._extract_image_embedding(image2)
                else:
                    emb = np.zeros(expected_dim, dtype=np.float32)

                assert emb.shape[0] == expected_dim, f"Image2 embedding shape {emb.shape} does not match expected dimension {expected_dim}"

            elif modality == "text":
                if modality in use_modalities and tokenized_text is not None:
                    emb = self._extract_text_embedding(tokenized_text)
                else:
                    emb = np.zeros(expected_dim, dtype=np.float32)
            else:
                raise ValueError(f"Modality '{modality}' not available in index. Available: {self.modalities}")

            # Ensure 1D - handle batch dimension
            if emb.ndim == 2:
                if emb.shape[0] == 1:
                    emb = emb[0]
                else:
                    # Raise error if batch size > 1
                    raise ValueError(
                        f"{modality} embedding has batch size {emb.shape[0]}, "
                        f"expected single embedding (batch size 1 or no batch dimension). "
                        f"Shape: {emb.shape}"
                    )
            elif emb.ndim > 2:
                raise ValueError(f"{modality} embedding has unexpected shape: {emb.shape}, expected 1D or 2D")

            # Ensure it's 1D now
            if emb.ndim != 1:
                raise ValueError(f"{modality} embedding is not 1D after processing: shape {emb.shape}")

            # Truncate or pad to match expected dimension
            assert emb.shape[0] == expected_dim, f"{modality} embedding shape {emb.shape} does not match expected dimension {expected_dim}"
            embeddings_list.append(emb)

        # Ensure all embeddings are 1D before concatenation
        for i, emb in enumerate(embeddings_list):
            if emb.ndim != 1:
                raise ValueError(f"Embedding at index {i} has shape {emb.shape}, expected 1D array")

        # Concatenate all parts
        query_embedding = np.concatenate(embeddings_list, axis=0)  # Use axis=0 for 1D arrays

        # Verify dimension matches index
        expected_dim = sum(self.embedding_dims[mod] for mod in self.modalities)
        if query_embedding.shape[0] != expected_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: got {query_embedding.shape[0]}, "
                f"expected {expected_dim}. "
                f"Modalities: {self.modalities}, "
                f"Embedding dims: {self.embedding_dims}, "
                f"Use modalities: {use_modalities}, "
                f"Individual embedding shapes: {[emb.shape for emb in embeddings_list]}"
            )

        # Normalize for cosine similarity
        query_embedding = query_embedding.astype(np.float32)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        return query_embedding

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Search for top-k nearest neighbors.

        Args:
            query_embedding: Query embedding array (total_embed_dim,)
            k: Number of nearest neighbors to retrieve

        Returns:
            Tuple of (distances, indices, metadata_list)
            - distances: Cosine similarity scores (k,)
            - indices: Indices of nearest neighbors (k,)
            - metadata_list: List of metadata dicts for the neighbors
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[None, ...]  # Add batch dimension

        # Verify dimension matches index before searching
        query_dim = query_embedding.shape[1]
        index_dim = self.index.d
        if query_dim != index_dim:
            raise ValueError(
                f"Query embedding dimension ({query_dim}) does not match index dimension ({index_dim}). "
                f"Query shape: {query_embedding.shape}, Index dimension: {self.index.d}"
            )

        # Search
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)

        # Flatten results (remove batch dimension)
        distances = distances[0]
        indices = indices[0]

        # Get metadata for results
        metadata_list = [self.metadata[idx] for idx in indices]

        return distances, indices, metadata_list

    def fetch_neighbors(
        self,
        observation,
        use_modalities: Optional[List[str]] = None,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Fetch top-k nearest neighbors for a query observation.

        Args:
            observation: Observation object/dict with images and text
            use_modalities: List of modalities to use. If None, uses all available.
            k: Number of nearest neighbors to retrieve

        Returns:
            Tuple of (distances, indices, metadata_list)
        """
        query_emb = self.create_query_embedding(
            observation=observation,
            use_modalities=use_modalities,
        )

        return self.search(query_emb, k=k)

