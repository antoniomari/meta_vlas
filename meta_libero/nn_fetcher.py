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
            self.modalities = list(meta["modalities"])
            self.embedding_dims = dict(meta["embedding_dims"])
            self.total_samples = meta["total_samples"]
            # Load normalize_per_modality flag (default to False for backward compatibility)
            self.normalize_per_modality = meta.get("normalize_per_modality", False)

        print(f"Loaded index with {self.index.ntotal} samples")
        print(f"Modalities: {self.modalities}")
        print(f"Embedding dimensions: {self.embedding_dims}")
        print(f"Normalize per modality: {self.normalize_per_modality}")
        self._modality_offsets = self._build_modality_offsets()
        self._expected_dim = sum(self.embedding_dims[mod] for mod in self.modalities)
        self._zero_embeddings = {
            mod: np.zeros(self.embedding_dims[mod], dtype=np.float32) for mod in self.modalities
        }
        self._all_vectors_cache: Optional[np.ndarray] = None
        self._all_text_cache: Optional[np.ndarray] = None
        self._all_non_text_cache: Dict[str, np.ndarray] = {}

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

    def _build_modality_offsets(self) -> Dict[str, Tuple[int, int]]:
        offsets: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for mod in self.modalities:
            dim = self.embedding_dims[mod]
            offsets[mod] = (offset, offset + dim)
            offset += dim
        return offsets

    @staticmethod
    def _normalize_if_needed(emb: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        return emb.astype(np.float32, copy=False)

    def _ensure_all_vectors_cache(self) -> np.ndarray:
        """Lazily cache reconstructed index vectors for repeated text-first queries."""
        if self._all_vectors_cache is None:
            n = self.index.ntotal
            all_vectors = self.index.reconstruct_n(0, n).astype(np.float32, copy=False)
            self._all_vectors_cache = all_vectors
            if "text" in self._modality_offsets:
                s, e = self._modality_offsets["text"]
                self._all_text_cache = all_vectors[:, s:e]
            self._all_non_text_cache = {
                mod: all_vectors[:, s:e]
                for mod, (s, e) in self._modality_offsets.items()
                if mod != "text"
            }
        assert self._all_vectors_cache is not None
        return self._all_vectors_cache

    def _extract_query_inputs(self, observation) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        image1 = None
        image2 = None
        tokenized_text = None

        if hasattr(observation, "images"):
            if "base_0_rgb" in observation.images:
                img = np.asarray(observation.images["base_0_rgb"])
                if img.ndim == 4:
                    if img.shape[0] > 1:
                        raise ValueError(
                            f"Image1 has batch size {img.shape[0]}, expected single image (batch size 1 or no batch dimension)"
                        )
                    img = img[0]
                image1 = img
            if "left_wrist_0_rgb" in observation.images:
                img = np.asarray(observation.images["left_wrist_0_rgb"])
                if img.ndim == 4:
                    if img.shape[0] > 1:
                        raise ValueError(
                            f"Image2 has batch size {img.shape[0]}, expected single image (batch size 1 or no batch dimension)"
                        )
                    img = img[0]
                image2 = img
            if hasattr(observation, "tokenized_prompt") and observation.tokenized_prompt is not None:
                txt = np.asarray(observation.tokenized_prompt)
                if txt.ndim == 2:
                    if txt.shape[0] > 1:
                        raise ValueError(
                            f"Tokenized text has batch size {txt.shape[0]}, expected single text (batch size 1 or no batch dimension)"
                        )
                    txt = txt[0]
                tokenized_text = txt
        elif isinstance(observation, dict):
            if "observation/image" in observation:
                img = np.asarray(observation["observation/image"])
                if img.ndim == 4:
                    if img.shape[0] > 1:
                        raise ValueError(
                            f"Image1 has batch size {img.shape[0]}, expected single image (batch size 1 or no batch dimension)"
                        )
                    img = img[0]
                image1 = img
            if "observation/wrist_image" in observation:
                img = np.asarray(observation["observation/wrist_image"])
                if img.ndim == 4:
                    if img.shape[0] > 1:
                        raise ValueError(
                            f"Image2 has batch size {img.shape[0]}, expected single image (batch size 1 or no batch dimension)"
                        )
                    img = img[0]
                image2 = img

        return image1, image2, tokenized_text

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
        embeddings = np.asarray(embeddings_jax, dtype=np.float32)

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
        tokenized_text_jax = (
            jnp.asarray(tokenized_text.astype(np.int32))
            if isinstance(tokenized_text, np.ndarray)
            else tokenized_text
        )

        # Call JIT function directly
        embedding_jax = self.text_embedding_fn(tokenized_text_jax)
        embedding = np.asarray(embedding_jax, dtype=np.float32)

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
        use_modalities_set = set(use_modalities)

        # Validate modalities
        for mod in use_modalities:
            if mod not in self.modalities:
                raise ValueError(f"Modality '{mod}' not available in index. Available: {self.modalities}")

        image1, image2, tokenized_text = self._extract_query_inputs(observation)

        # Build embedding parts in order matching the index
        # Simple logic: for each modality, if it's in use_modalities and data is available, extract it; otherwise use zeros
        embeddings_list = []

        for modality in self.modalities:
            expected_dim = self.embedding_dims[modality]

            if modality == "image1":
                if modality in use_modalities_set and image1 is not None:
                    emb = self._extract_image_embedding(image1)
                    if self.normalize_per_modality:
                        emb = self._normalize_if_needed(emb)
                else:
                    emb = self._zero_embeddings["image1"]

                assert emb.shape[0] == expected_dim, f"Image1 embedding shape {emb.shape} does not match expected dimension {expected_dim}"

            elif modality == "image2":
                if modality in use_modalities_set and image2 is not None:
                    emb = self._extract_image_embedding(image2)
                    if self.normalize_per_modality:
                        emb = self._normalize_if_needed(emb)
                else:
                    emb = self._zero_embeddings["image2"]

                assert emb.shape[0] == expected_dim, f"Image2 embedding shape {emb.shape} does not match expected dimension {expected_dim}"

            elif modality == "text":
                if modality in use_modalities_set and tokenized_text is not None:
                    emb = self._extract_text_embedding(tokenized_text)
                    if self.normalize_per_modality:
                        emb = self._normalize_if_needed(emb)
                else:
                    emb = self._zero_embeddings["text"]
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
        if query_embedding.shape[0] != self._expected_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: got {query_embedding.shape[0]}, "
                f"expected {self._expected_dim}. "
                f"Modalities: {self.modalities}, "
                f"Embedding dims: {self.embedding_dims}, "
                f"Use modalities: {use_modalities}, "
                f"Individual embedding shapes: {[emb.shape for emb in embeddings_list]}"
            )

        # Normalize for cosine similarity (skip if per-modality normalization was used)
        query_embedding = query_embedding.astype(np.float32, copy=False)
        if not self.normalize_per_modality:
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

    def search_text_then_images(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        text_similarity_threshold: float = 0.99,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Two-stage search: first filter by text similarity (~1.0), then rank by image similarity.

        Requires per-modality normalization and that the index contains image1, image2, and text.
        The concatenated vector is [image1_emb | image2_emb | text_emb], each individually normalized.
        We reconstruct all index vectors, compute per-modality dot products, filter by text, rank by images.

        Args:
            query_embedding: Full query embedding (total_embed_dim,)
            k: Number of nearest neighbors to return
            text_similarity_threshold: Minimum text cosine similarity to keep (default 0.99)

        Returns:
            Tuple of (image_similarities, indices, metadata_list)
            - image_similarities: Combined image1+image2 similarity scores (k,)
            - indices: Indices of nearest neighbors (k,)
            - metadata_list: List of metadata dicts for the neighbors
        """
        query_embedding = query_embedding.astype(np.float32, copy=False)

        if "text" not in self._modality_offsets:
            raise ValueError("text modality required for text_then_images search")

        # Slice query into modality parts
        text_start, text_end = self._modality_offsets["text"]
        query_text = query_embedding[text_start:text_end]

        # Reconstruct all vectors from the index (cached).
        self._ensure_all_vectors_cache()

        # Compute text similarity for all vectors
        assert self._all_text_cache is not None
        text_sims = self._all_text_cache @ query_text  # (n,) — cosine similarity for normalized vectors

        # Filter: keep only samples with text similarity >= threshold
        mask = text_sims >= text_similarity_threshold
        candidate_indices = np.where(mask)[0]

        print(f"Found {len(candidate_indices)} candidates")

        if len(candidate_indices) == 0:
            # Fallback: relax threshold and take top candidates by text similarity
            top_text = np.argsort(-text_sims)[:k]
            candidate_indices = top_text

        image_sims = np.zeros(len(candidate_indices), dtype=np.float32)

        for mod in self.modalities:
            if mod == "text":
                continue
            mod_start, mod_end = self._modality_offsets[mod]
            query_mod = query_embedding[mod_start:mod_end]
            candidate_mod = self._all_non_text_cache[mod][candidate_indices]
            image_sims += candidate_mod @ query_mod  # Add per-modality cosine sim

        # Rank by combined image similarity (descending)
        top_k = min(k, len(candidate_indices))
        best = np.argsort(-image_sims)[:top_k]

        result_indices = candidate_indices[best]
        result_sims = image_sims[best]
        metadata_list = [self.metadata[idx] for idx in result_indices]

        return result_sims, result_indices, metadata_list

    def fetch_neighbors(
        self,
        observation,
        use_modalities: Optional[List[str]] = None,
        k: int = 10,
        filter_text_first: bool = False,
        text_similarity_threshold: float = 0.999999,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Fetch top-k nearest neighbors for a query observation.

        Args:
            observation: Observation object/dict with images and text
            use_modalities: List of modalities to use. If None, uses all available.
            k: Number of nearest neighbors to retrieve
            filter_text_first: If True, first filter by text similarity (~1.0),
                               then rank remaining by image similarity only.
            text_similarity_threshold: Min text cosine similarity when filter_text_first=True.

        Returns:
            Tuple of (distances, indices, metadata_list)
        """
        query_emb = self.create_query_embedding(
            observation=observation,
            use_modalities=use_modalities,
        )

        if filter_text_first:
            return self.search_text_then_images(
                query_emb, k=k, text_similarity_threshold=text_similarity_threshold,
            )

        return self.search(query_emb, k=k)

