## Training Function
# Suppress all warnings at the very beginning
import warnings
import contextlib
import copy
import traceback
import dataclasses
import functools
from typing import Any, Iterator, SupportsIndex, Tuple, Optional, List
import os
import logging
import etils.epath as epath
import time
import collections
import math
import pathlib
import imageio
from PIL import Image, ImageDraw, ImageFont
import sys
import random
import h5py
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
import openpi.models.pi0_config as pi0_config
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

sys.path.append("third_party/libero")
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi.policies import policy_config as _policy_config
from openpi.policies import policy as _policy
from openpi.training.data_loader import (
    Dataset,
    DataLoader,
    FakeDataset,
    TransformedDataset,
    TorchDataLoader,
    DataLoaderImpl,
    transform_dataset,
)
import openpi.transforms as _transforms
import openpi.transforms as transforms
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import matplotlib.pyplot as plt

import tqdm
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils
import openpi.shared.download as download

import torch
# Ensures deterministic behavior in CUDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set python hash seed
os.environ["PYTHONHASHSEED"] = str(12345)

# To fix versioning issues with torch
torch.serialization.add_safe_globals(
    [
        np.core.multiarray._reconstruct,  # noqa
        np.ndarray,
        np.dtype,
        np.dtypes.Float64DType,
    ]
)


def decode_tokenized_prompt(
    tokenized_prompt: np.ndarray | jnp.ndarray | None,
    tokenized_prompt_mask: np.ndarray | jnp.ndarray | None = None,
    max_token_len: int = 200,
    fallback_text: str = "Task",
) -> str:
    """
    Decode a tokenized prompt from an observation to human-readable text.

    Args:
        tokenized_prompt: Tokenized prompt array, shape (seq_len,) or (batch, seq_len).
                          If None, returns fallback_text.
        tokenized_prompt_mask: Optional mask array indicating valid tokens, same shape as tokenized_prompt.
                               If provided, only tokens where mask is True are decoded.
        max_token_len: Maximum token length for tokenizer initialization.
        fallback_text: Text to return if decoding fails or tokenized_prompt is None.

    Returns:
        Decoded text string. Returns fallback_text if decoding fails or input is None.
    """
    if tokenized_prompt is None:
        return fallback_text

    try:
        from openpi.models import tokenizer as _tokenizer

        # Create tokenizer instance
        tok = _tokenizer.PaligemmaTokenizer(max_len=max_token_len)

        # Extract tokens (remove padding based on mask if available)
        tokens = np.array(tokenized_prompt)

        # Handle both 1D and 2D token arrays
        if len(tokens.shape) > 1:
            tokens = tokens.flatten()

        # Extract valid tokens using mask if available
        if tokenized_prompt_mask is not None:
            mask = np.array(tokenized_prompt_mask)
            # Handle both 1D and 2D mask arrays
            if len(mask.shape) > 1:
                mask = mask.flatten()
            # Get only the valid tokens (where mask is True)
            valid_tokens = tokens[mask].tolist()
        else:
            # Remove padding (zeros) manually
            valid_tokens = [int(t) for t in tokens if int(t) != 0]

        # Decode tokens to text
        if not valid_tokens:
            return fallback_text

        decoded_text = tok._tokenizer.decode(valid_tokens)

        # Clean up the decoded text
        decoded_text = decoded_text.strip()
        # Remove common prefixes/suffixes that might be added during tokenization
        if decoded_text.startswith("<bos>"):
            decoded_text = decoded_text[5:].strip()

        return decoded_text

    except Exception as e:
        logging.warning(f"Failed to decode tokenized prompt: {e}")
        return fallback_text


def _observation_batch_size(observation: _model.Observation) -> int:
    """Return batch size from observation (1 if single observation)."""
    img = observation.images.get("base_0_rgb")
    if img is None:
        img = observation.images.get("left_wrist_0_rgb")
    if img is None:
        return 1
    arr = np.array(img)
    if arr.ndim >= 4 and arr.shape[0] > 1:
        return int(arr.shape[0])
    return 1


def plot_observation_with_decoded_prompt(
    observation: _model.Observation,
    similarity_score: float | np.ndarray | jnp.ndarray | None = None,
    current_task_description: str | None = None,
    max_token_len: int = 200,
    plot_title_prefix: str = "Observation",
) -> None:
    """
    Plot observation(s) with decoded task description from tokenized prompt.

    If observation is batched (tensors have leading batch dimension), plots a grid:
    one row per observation, two columns (base camera, wrist camera). similarity_score
    may be a scalar or an array of length batch_size.

    Args:
        observation: Observation (single or batched) with images and optional tokenized_prompt.
        similarity_score: Optional score(s); scalar or array of length batch_size.
        current_task_description: Fallback task description if decoding fails.
        max_token_len: Maximum token length for tokenizer initialization.
        plot_title_prefix: Prefix for the plot title.
    """
    batch_size = _observation_batch_size(observation)
    scores = np.atleast_1d(similarity_score) if similarity_score is not None else np.full(batch_size, np.nan)
    if scores.size == 1 and batch_size > 1:
        scores = np.broadcast_to(scores, (batch_size,))
    scores = np.asarray(scores).flat[:batch_size]

    images1 = observation.images.get("base_0_rgb")
    images2 = observation.images.get("left_wrist_0_rgb")
    assert images1 is not None and images2 is not None

    fig, axes = plt.subplots(batch_size, 2, figsize=(8, 4 * batch_size), squeeze=False)

    for i in range(batch_size):
        # Decode prompt
        token_prompt = observation.tokenized_prompt
        token_mask = observation.tokenized_prompt_mask
        assert token_prompt is not None
        tp = np.array(token_prompt)
        tm = np.array(token_mask) if token_mask is not None else None
        token_prompt_i = tp[i]
        token_mask_i = tm[i] if tm is not None else None

        decoded_task_description = decode_tokenized_prompt(
            tokenized_prompt=token_prompt_i,
            tokenized_prompt_mask=token_mask_i,
            max_token_len=max_token_len,
            fallback_text=current_task_description or "Task",
        )

        title_parts = [plot_title_prefix]
        if i < len(scores) and not np.isnan(scores[i]):
            title_parts.append(f"(similarity: {scores[i]:.4f})")
        if batch_size > 1:
            title_parts.append(f"[{i}]")
        title_parts.append(f": {decoded_task_description}")
        row_title = " ".join(title_parts)

        # NOTE: need to rescale in the right range here
        # they are in the range [-1, 1]
        # need to rescale to [0, 1]
        img1 = (images1[i] + 1.0) / 2.0 # render_image(images1[i])
        img2 = (images2[i] + 1.0) / 2.0 # render_image(images2[i])

        for col, (img, cam_name) in enumerate(
            [(img1, "Base Camera"), (img2, "Wrist Camera")]
        ):
            ax = axes[i, col]
            assert img is not None
            ax.imshow(img)
            ax.set_title(decoded_task_description, fontsize=10)
            ax.axis("off")

        axes[i, 0].set_ylabel(row_title, fontsize=9, labelpad=4)

    fig.suptitle(plot_title_prefix, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def make_observation_from_simulator(
    policy: _policy.Policy, curr_obs_dict: dict
) -> _model.Observation:
    # Create Observation object from the current obs_dict
    inputs = jax.tree.map(lambda x: x, curr_obs_dict)
    inputs = policy._input_transform(inputs)
    if not policy._is_pytorch_model:
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    else:
        # Convert inputs to PyTorch tensors and move to correct device
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(policy._pytorch_device)[
                None, ...
            ],
            inputs,
        )

    observation = _model.Observation.from_dict(inputs)

    return observation


def _draw_step_on_frame(frame: np.ndarray, step: int) -> np.ndarray:
    """Draw step number on top of a video frame (camera image). Returns a copy with text overlaid."""
    out = np.asarray(frame).copy()
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()
    text = f"Step: {step}"
    pad = 6
    # Fixed-size background rectangle for readability (avoids textbbox on older Pillow)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        w, h = max(60, len(text) * 10), 28
    draw.rectangle(
        [(0, 0), (w + 2 * pad, h + 2 * pad)],
        fill=(0, 0, 0),
        outline=(255, 255, 255),
    )
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return np.array(pil)


def render_image(img: np.ndarray, resize_size: int = 224) -> np.ndarray:
    """
    Render an image from the environment. Rotate 180 degrees and resize to the given size, convert to uint8.
    Args:
        img: The image to render.
        resize_size: The size to resize the image to.

    Returns:
        The rendered image.
    """
    img = np.ascontiguousarray(img[::-1, ::-1])
    img_resized = image_tools.resize_with_pad(img, resize_size, resize_size)
    img = image_tools.convert_to_uint8(img_resized)
    return img
