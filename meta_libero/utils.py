## Training Function
# Suppress all warnings at the very beginning
import warnings

warnings.filterwarnings("ignore")
# Specifically suppress the JAX shape deprecation warning from Flax
warnings.filterwarnings(
    "ignore", message=".*shape requires ndarray or scalar arguments.*"
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax.core.scope")

import contextlib
import copy
import traceback
import dataclasses
import functools
from typing import Any, Iterator, SupportsIndex, Tuple, Optional, List
import os
import logging

# Enable JAX compilation logging to see when recompilation happens

# Configure logging: suppress verbose JAX/absl warnings but keep compilation logs
logging.getLogger("absl").setLevel(logging.WARNING)  # Show warnings but not all info
import etils.epath as epath
import time

# Evaluate pretrained model on first task of libero_90
import collections
import math
import pathlib
import imageio
import sys
import random
import h5py

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops"

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

import tqdm

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax

try:
    from tqdm.auto import tqdm  # Use auto for notebook compatibility
except ImportError:
    from tqdm import tqdm

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

PRINT_MEMORY_CHECKPOINT = False
PRINT_TIME_EXECUTION = False


# Memory tracking utilities
def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    try:
        import jax

        devices = jax.devices()
        if not devices:
            return None
        # Get memory info from the first device
        mem_info = devices[0].memory_stats()
        allocated = mem_info.get("bytes_in_use", 0) / (1024**3)  # Convert to GB
        reserved = mem_info.get("bytes_reserved", 0) / (1024**3)  # Convert to GB
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": reserved,  # Total reserved is usually the peak
        }
    except Exception as e:
        return {"error": str(e)}


def print_memory_checkpoint(label: str, line_num: int = None):
    """Print memory usage at a checkpoint."""
    if PRINT_MEMORY_CHECKPOINT:
        mem = get_gpu_memory_usage()
        if mem and "error" not in mem:
            line_info = f" (line {line_num})" if line_num else ""
            print(
                f"[MEMORY CHECKPOINT{line_info}] {label}: "
                f"Allocated: {mem['allocated_gb']:.2f} GB, "
                f"Reserved: {mem['reserved_gb']:.2f} GB"
            )
        else:
            print(f"[MEMORY CHECKPOINT] {label}: {mem}")


# Global flag to disable JIT (set to True to disable JIT compilation)
DISABLE_JIT = False


def check_equality(state_a, state_b):
    # Flatten both states and check if all arrays are identical
    leaves_a = jax.tree_util.tree_leaves(state_a.params)
    leaves_b = jax.tree_util.tree_leaves(state_b.params)

    all_match = all(jnp.array_equal(a, b) for a, b in zip(leaves_a, leaves_b))
    return all_match


import matplotlib.pyplot as plt


def plot_observation_images(observation, task_description=None, dataset_dir=None):
    """
    Plot the two images from a retrieved observation.

    Args:
        observation: Either:
            - An Observation object with 'images' attribute containing 'base_0_rgb' and 'left_wrist_0_rgb'
            - A dict with metadata containing 'file_name', 'demo_key', 'camera_view', 'step_idx' to load from HDF5
        task_description: Task description string (if None, will try to extract from observation)
        dataset_dir: Path to dataset directory (required if observation is metadata dict)
    """
    # Extract images
    if hasattr(observation, "images"):
        # It's an Observation object
        image1 = observation.images.get("base_0_rgb")
        image2 = observation.images.get("left_wrist_0_rgb")

        # Convert to numpy if needed
        if image1 is not None:
            image1 = np.array(image1)
        if image2 is not None:
            image2 = np.array(image2)

        # Get task description if available
        if task_description is None and hasattr(observation, "tokenized_prompt"):
            # Try to get from prompt if available (would need to decode, but for now just use placeholder)
            task_description = "Task"
    elif isinstance(observation, dict):
        # It's metadata dict - load from HDF5
        if dataset_dir is None:
            raise ValueError(
                "dataset_dir is required when observation is a metadata dict"
            )

        file_path = pathlib.Path(dataset_dir) / observation["file_name"]
        with h5py.File(file_path, "r") as f:
            demo_data = f["data"][observation["demo_key"]]

            # Load base image (base_0_rgb)
            if "base_0_rgb" in demo_data["obs"]:
                image1 = np.array(
                    demo_data["obs"]["base_0_rgb"][observation["step_idx"]]
                )
            elif "camera_view" in observation:
                image1 = np.array(
                    demo_data["obs"][observation["camera_view"]][
                        observation["step_idx"]
                    ]
                )
            else:
                image1 = None

            # Load wrist image (left_wrist_0_rgb)
            if "left_wrist_0_rgb" in demo_data["obs"]:
                image2 = np.array(
                    demo_data["obs"]["left_wrist_0_rgb"][observation["step_idx"]]
                )
            else:
                image2 = None

        # Get task description from metadata if available
        if task_description is None:
            task_description = observation.get("task_description", "Task")
    else:
        raise ValueError(f"Unsupported observation type: {type(observation)}")

    # Normalize images to [0, 1] range
    def normalize_image(img):
        if img is None:
            return None
        img = np.array(img)
        # Remove batch dimension if present (shape like (1, H, W, C) -> (H, W, C))
        while len(img.shape) == 4 and img.shape[0] == 1:
            img = img[0]
        # Ensure image is 2D (grayscale) or 3D (RGB), not 4D
        if len(img.shape) not in [2, 3]:
            raise ValueError(
                f"Invalid image shape after removing batch dimension: {img.shape}"
            )
        if img.max() > 1.0:
            img = img / 255.0
        elif img.max() <= 1.0 and img.dtype != np.uint8:
            # Already normalized
            pass
        return np.clip(img, 0, 1)

    image1_norm = normalize_image(image1) if image1 is not None else None
    image2_norm = normalize_image(image2) if image2 is not None else None

    # Create plot with 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(task_description or "Task", fontsize=12, fontweight="bold")

    # Plot image1 (base camera) - flip vertically to correct orientation
    if image1_norm is not None:
        axes[0].imshow(np.flipud(image1_norm))
        axes[0].set_title("Base Camera", fontsize=10)
    else:
        axes[0].text(
            0.5, 0.5, "No image", ha="center", va="center", transform=axes[0].transAxes
        )
        axes[0].set_title("Base Camera (missing)", fontsize=10)
    axes[0].axis("off")

    # Plot image2 (wrist camera)
    if image2_norm is not None:
        axes[1].imshow(np.flipud(image2_norm))
        axes[1].set_title("Wrist Camera", fontsize=10)
    else:
        axes[1].text(
            0.5, 0.5, "No image", ha="center", va="center", transform=axes[1].transAxes
        )
        axes[1].set_title("Wrist Camera (missing)", fontsize=10)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)  # Explicitly close figure to prevent memory leaks


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


def plot_observation_with_decoded_prompt(
    observation: _model.Observation,
    similarity_score: float | None = None,
    current_task_description: str | None = None,
    max_token_len: int = 200,
    plot_title_prefix: str = "Observation",
) -> None:
    """
    Plot an observation's images with its decoded task description from tokenized prompt.

    This function extracts the task description from the observation's tokenized_prompt
    by decoding it, then plots the observation images with the decoded description.

    Args:
        observation: Observation object containing images and optional tokenized_prompt.
        similarity_score: Optional similarity score to include in the plot title.
        current_task_description: Fallback task description if decoding fails.
        max_token_len: Maximum token length for tokenizer initialization.
        plot_title_prefix: Prefix for the plot title (e.g., "Current Observation" or "Fetched Observation").
    """
    # Decode task description from tokenized prompt
    decoded_task_description = decode_tokenized_prompt(
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        max_token_len=max_token_len,
        fallback_text=current_task_description or "Task",
    )

    # Build plot title
    title_parts = [plot_title_prefix]
    if similarity_score is not None:
        title_parts.append(f"(similarity: {similarity_score:.4f})")
    title_parts.append(f": {decoded_task_description}")
    plot_title = " ".join(title_parts)

    # Plot the observation
    plot_observation_images(
        observation=observation,
        task_description=plot_title,
        dataset_dir=None,  # Not needed since we have Observation object
    )


def fetch_samples(dataset, idx) -> Tuple[_model.Observation, _model.Actions]:
    examples = [dataset[int(i)] for i in idx]
    batch = _data_loader._collate_fn(examples)

    # Convert to JAX arrays (model expects JAX arrays, not numpy arrays)
    def to_jax(x):
        if isinstance(x, torch.Tensor):
            return jnp.asarray(x)
        elif isinstance(x, np.ndarray):
            return jnp.asarray(x)
        elif isinstance(x, jax.Array):
            return x
        else:
            return jnp.asarray(x)

    batch = jax.tree.map(to_jax, batch)
    obs, actions = _model.Observation.from_dict(batch), batch["actions"]
    return obs, actions


def load_pi05_libero_model(
    use_lora: bool = False, action_expert_only: bool = False
) -> Tuple[_model.BaseModel, _config.TrainConfig]:
    # Load weights from pi0.5 model already fine-tuned on Libero
    checkpoint_gs_path = "gs://openpi-assets/checkpoints/pi05_libero"
    checkpoint_name = "pi05_libero" if not use_lora else "pi05_libero_lora"
    config = _config.get_config(checkpoint_name)

    # Download checkpoint
    checkpoint_dir = download.maybe_download(checkpoint_gs_path)
    # Load model
    model = config.model.load(
        _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    )

    if action_expert_only:
        print("Training action expert only")
        freeze_filter = pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ).get_freeze_filter_action_expert()
        config = dataclasses.replace(config, freeze_filter=freeze_filter)

    return model, config


@at.typecheck
@functools.partial(
    jax.jit, static_argnums=(0,), donate_argnums=(2,)
)  # config is static (Python object, not JAX array) - use argnums instead of argnames
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    # Memory checkpoint: start of train_step
    # NOTE: Removed all jax.lax.cond print statements to avoid recompilation
    # When step_int changes (0, 1, 2, then >= 3), JAX sees different execution paths
    # and recompiles the function, causing 42s delay on every step
    # Debug prints should be done outside the JIT function instead

    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)

    # NOTE: here goes out of memory 31.42 GB -> OOM ()
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)

    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_params, opt_state=new_opt_state
    )
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(
                nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")
            ),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }

    return new_state, info


@functools.partial(
    jax.jit, static_argnums=(0, 1, 2, 3), donate_argnums=(5,)
)  # config, model, weight_decay, learning_rate, warmup_steps are static (Python object, not JAX array) - use argnums instead of argnames
def init_train_state(
    config: _config.TrainConfig,
    weight_decay: float,
    learning_rate: float,
    warmup_steps: int,
    graphdef: nnx.GraphDef,
    params: at.Params,
) -> training_utils.TrainState:
    """Initialize train state from params. This function will be JIT-compiled."""
    # Get graphdef for the model (static structure)
    # graphdef = nnx.graphdef(model)

    # Initialize optimizer with cosine decay schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate / (warmup_steps + 1),
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=30000,
        end_value=learning_rate * 0.1,
    )

    # Create optimizer
    if weight_decay > 0:
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(),
            optax.add_decayed_weights(weight_decay),
            optax.scale_by_schedule(lr_schedule),
            optax.scale(-1.0),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(),
            optax.scale_by_schedule(lr_schedule),
            optax.scale(-1.0),
        )

    # Apply freeze filter if provided
    print_memory_checkpoint("Before applying freeze filter", 615)
    print_memory_checkpoint("After applying freeze filter", 623)

    # Initialize optimizer state on trainable params
    opt_state = tx.init(params.filter(config.trainable_filter))

    print_memory_checkpoint("After initializing optimizer state", 624)

    # Create and return TrainState
    # Following marco.py pattern: ema_params references same params initially
    return training_utils.TrainState(
        step=0,
        params=params,
        model_def=graphdef,
        tx=tx,
        opt_state=opt_state,
        ema_decay=config.ema_decay,
        ema_params=(
            None if config.ema_decay is None else params
        ),  # This is creating a copy
    )


def print_trainable_parameters(params: at.Params, trainable_filter):
    def param_count(x):
        if hasattr(x, "value"):
            return int(jax.device_get(jnp.size(x.value)))
        return int(jax.device_get(jnp.size(x)))

    param_counts = jax.tree.map(
        param_count, params.filter(trainable_filter)
    )
    num_trainable_params = sum(jax.tree_util.tree_leaves(param_counts))
    print(f"Number of trainable parameters: {num_trainable_params}")
    del param_counts


def train_model_on_fly(
    model: _model.BaseModel,
    training_data_loader: _data_loader.DataLoader,
    config: _config.TrainConfig,
    learning_rate: float = 2.5e-5,
    num_steps: int = 1000,
    warmup_steps: int = 100,
    weight_decay: float = 0.0,
    log_interval: int = 100,
    seed: int = 42,
    # Resume parameters - pass these to continue training from a previous run
    resume_train_state: training_utils.TrainState | None = None,
    resume_losses: list[float] | None = None,
    # Control buffer donation - disable for TTT since we extract model immediately after
    donate_buffers: bool = True,
) -> tuple[_model.BaseModel, list[float], training_utils.TrainState]:
    """
    Train a model on the fly and return a copy of the trained model, training losses, and final train state.

    Args:
        model: The model to train (will be copied internally, ignored if resume_train_state is provided)
        training_data_loader: _data_loader.DataLoader,
        learning_rate: Learning rate for optimizer
        num_steps: Number of additional gradient steps to perform (not total steps)
        batch_size: Batch size for training
        warmup_steps: Number of warmup steps for learning rate schedule
        weight_decay: Weight decay coefficient
        trainable_filter: Filter for trainable parameters (None = all trainable)
        freeze_filter: Filter for frozen parameters (None = none frozen)
        log_interval: Log training info every N steps
        seed: Random seed
        resume_train_state: Optional TrainState to resume from. If provided, will continue training from this state.
        resume_losses: Optional list of previous losses to continue from.

    Returns:
        A tuple of (trained_model, losses, train_state) where:
            - trained_model: The trained model
            - losses: List of loss values for each step (including previous if resumed)
            - train_state: Final TrainState (can be passed to resume_train_state for continuation)
    """
    print_memory_checkpoint("train_model_on_fly: START", 452)

    # Seed all random number generators for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Set the JAX compilation cache directory to avoid recompilation and speed up repeated runs.
    jax.config.update(
        "jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser())
    )

    # Speed up JIT compilation by reducing XLA autotuning (faster compilation, minimal performance impact)
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
    elif "--xla_gpu_autotune_level" not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0"
        )

    # Determine trainable filter
    _trainable_filter_for_init = config.trainable_filter
    # if config.trainable_filter is not None:
    #    _trainable_filter_for_init = config.trainable_filter
    # elif config.freeze_filter is not None:
    #    _trainable_filter_for_init = nnx.All(nnx.Param, nnx.Not(config.freeze_filter))
    # else:
    #    _trainable_filter_for_init = nnx.Param

    # Check if resuming from a previous train state
    if resume_train_state is not None:
        # Use the provided train state directly
        train_state = resume_train_state
        start_step = int(train_state.step)
        end_step = start_step + num_steps
        print(
            f"Resuming training from step {start_step}, will train {num_steps} more steps to step {end_step}"
        )

        # Count trainable parameters from the resumed state
        trainable_params = train_state.params.filter(_trainable_filter_for_init)

        def param_count(x):
            if hasattr(x, "value"):
                return int(jax.device_get(jnp.size(x.value)))
            return int(jax.device_get(jnp.size(x)))

        param_counts = jax.tree.map(param_count, trainable_params)
        num_trainable_params = sum(jax.tree_util.tree_leaves(param_counts))
        del param_counts, trainable_params
        print(f"Number of trainable parameters: {num_trainable_params:,}")
    else:
        # Initialize fresh training state
        start_step = 0
        end_step = num_steps

        # Setup rng
        rng = jax.random.key(seed)
        train_rng, init_rng = jax.random.split(rng)
        initial_params = nnx.state(model)
        if config.freeze_filter is not None:
            initial_params = nnx_utils.state_map(
                initial_params,
                config.freeze_filter,
                lambda p: p.replace(p.value.astype(jnp.bfloat16)),
            )

        # JIT compile the initialization and call it with params
        # For TTT, we don't donate the model so it can be used for inference after training
        print_memory_checkpoint("Before JIT compilation of init_train_state", 560)

        graphdef = nnx.graphdef(model)
        train_state = init_train_state(
            config, weight_decay, learning_rate, warmup_steps, graphdef, initial_params
        )
        # NOTE: after this we go from 12.58 GB -> 25.13 GB why?
        print_memory_checkpoint("After JIT compilation of init_train_state", 566)

    # Initialize RNG
    rng = jax.random.key(seed)
    # Define trainable_filter for training step
    _trainable_filter = _trainable_filter_for_init

    # train_step is now automatically JIT-compiled via decorator
    # No need to create train_step_jit - just call train_step directly
    # JAX will automatically cache the compilation based on function signature and input shapes
    print_memory_checkpoint("train_step is JIT-compiled via decorator", 573)

    train_step_jit = functools.partial(train_step, config)
    print_memory_checkpoint("After setting up train_step", 585)

    # Check if compilation cache is enabled
    cache_dir = jax.config.jax_compilation_cache_dir
    # print(f"[DEBUG] JAX compilation cache directory: {cache_dir}")
    if cache_dir:
        cache_path = os.path.expanduser(cache_dir)
        if os.path.exists(cache_path):
            cache_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(cache_path)
                for filename in filenames
            ) / (
                1024**2
            )  # MB
            # print(f"[DEBUG] Compilation cache exists, size: {cache_size:.2f} MB")
        else:
            print(f"[DEBUG] Compilation cache directory does not exist yet")

    # Training loop
    # Initialize losses from resume_losses if provided, otherwise start fresh
    losses = list(resume_losses) if resume_losses else []
    infos = []

    # Disable progress bar for train_model_on_fly (called during TTT, progress shown in main evaluation loop)
    # Train from start_step to end_step (num_steps additional steps)
    pbar = tqdm(
        range(start_step, end_step),
        desc="Training",
        total=num_steps,  # Show progress as num_steps (the additional steps)
        dynamic_ncols=True,
        mininterval=0.5,
        maxinterval=2.0,
        disable=True,  # Disable progress bar for TTT training
    )

    # Initialize iterator
    data_iter = iter(training_data_loader)

    # Following marco.py: no separate warmup, first training step will compile naturally
    print("Starting training (first step will compile, may take a few minutes)...")
    print_memory_checkpoint("Before training loop starts", 604)

    for step in pbar:
        # Get batch from training set - aligned with train.py
        # TIMING: Measure how long it takes to fetch and move batch to GPU
        t0_fetch = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(training_data_loader)
            batch = next(data_iter)
        t1_fetch = time.perf_counter()
        batch_fetch_time = (t1_fetch - t0_fetch) * 1000  # Convert to ms
        if PRINT_TIME_EXECUTION:
            print(f"[TIMING] Batch fetch took {batch_fetch_time:.2f} ms")

        # Training step - aligned with train.py: pass rng, state, batch
        # TIMING: Measure training step (includes GPU computation)
        t0_before_call = time.perf_counter()
        if PRINT_TIME_EXECUTION:
            print(f"[TIMING] About to call train_step_jit at step {step}")

        t0_train = time.perf_counter()
        # JAX compilation logging is enabled via JAX_LOG_COMPILES=1
        # JAX will automatically log compilation events (cache hits/misses)

        # Call train_step directly (it's already JIT-compiled via decorator)
        train_state, info = train_step_jit(rng, train_state, batch)
        t1_after_call = time.perf_counter()
        call_time = (t1_after_call - t0_train) * 1000  # Time for JIT call to return

        if PRINT_TIME_EXECUTION:
            print(
                f"[TIMING] train_step_jit call returned in {call_time:.2f} ms ({call_time/1000:.2f} s), now blocking on loss..."
            )
            if (
                step > 0 and call_time > 5000
            ):  # More than 5 seconds on step > 0 suggests recompilation
                print(
                    f"[WARNING] Step {step} took {call_time/1000:.2f}s - possible cache miss/recompilation!"
                )

        t0_block = time.perf_counter()
        jax.block_until_ready(info["loss"])  # Wait for GPU to finish
        t1_block = time.perf_counter()
        block_time = (t1_block - t0_block) * 1000  # Time to block until ready

        t1_train = time.perf_counter()
        train_step_time = (t1_train - t0_train) * 1000  # Total time
        time_before_call = (t0_train - t0_before_call) * 1000  # Time before calling

        if PRINT_TIME_EXECUTION:
            print(f"[TIMING] Breakdown for step {step}:")
            print(f"  - Time before call: {time_before_call:.2f} ms")
            print(
                f"  - JIT call returned in: {call_time:.2f} ms ({call_time/1000:.2f} s)"
            )
            print(f"  - Block until ready: {block_time:.2f} ms")
            print(
                f"  - Total train_step: {train_step_time:.2f} ms ({train_step_time/1000:.2f} s)"
            )

        # Logging - aligned with train.py info structure
        loss_val = float(jax.device_get(info["loss"]))
        grad_norm = float(jax.device_get(info["grad_norm"]))
        losses.append(loss_val)  # Store loss for plotting
        infos.append({"loss": loss_val, "grad_norm": grad_norm})

        # Print timing info for every step using tqdm.write to avoid conflicts with progress bar
        total_time = batch_fetch_time + train_step_time
        fetch_percent = (batch_fetch_time / total_time) * 100
        pbar.write(
            f"Step {step}: fetch={batch_fetch_time:6.2f}ms ({fetch_percent:4.1f}%), train={train_step_time:6.2f}ms, loss={loss_val:.4f}"
        )
        # Note: Detailed timings are printed directly in train_step() when JIT is disabled

        # Update progress bar after every step
        pbar.set_postfix({"loss": f"{loss_val:.4f}", "grad_norm": f"{grad_norm:.4f}"})

        # Log timing for all steps in the progress bar
        if step == start_step:
            pbar.set_postfix(
                {"loss": f"{loss_val:.4f}", "time": f"{train_step_time/1000:.1f}s"}
            )
        else:
            pbar.set_postfix(
                {
                    "loss": f"{loss_val:.4f}",
                    "time": f"{train_step_time/1000:.1f}s",
                    "call": f"{call_time/1000:.1f}s",
                }
            )
        if step == start_step:
            print_memory_checkpoint(
                f"After first training step (step {step}) - compilation complete", 627
            )


        if step % log_interval == 0 or step == end_step - 1:
            avg_loss = np.mean([info["loss"] for info in infos[-log_interval:]])
            avg_grad_norm = np.mean(
                [info["grad_norm"] for info in infos[-log_interval:]]
            )
            pbar.write(f"Step {step}: loss={avg_loss:.4f}, grad_norm={avg_grad_norm:.4f}")

    # Merge to create a fresh model instance with independent buffers
    trained_model = nnx.merge(train_state.model_def, train_state.params)
    trained_model.eval()

    return trained_model, losses, train_state


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    """Initialize LIBERO environment."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def run_evaluation(
    policy: _policy.Policy,
    num_trials: int = 10,
    task_suite_name: str = "libero_90",
    task_id: int = 0,
    num_steps_wait: int = 10,
    save_video: bool = True,
    video_out_path: str = "data/libero/videos",
    task_description: str = "Task 0",
    seed: int = 0,
):
    """Run evaluation on a LIBERO task.

    Args:
        policy: The policy to evaluate. For reproducible results, create the policy with
                `create_policy(..., rng_seed=seed)` to initialize its internal RNG state.
        num_trials: Number of evaluation episodes to run.
        task_suite_name: Name of the LIBERO task suite (e.g., "libero_10", "libero_90").
        task_id: ID of the task within the suite to evaluate.
        num_steps_wait: Number of steps to wait for environment stabilization.
        save_video: Whether to save rollout videos.
        video_out_path: Directory path to save videos.
        task_description: Description of the task (overridden by actual task description).
        seed: Random seed for environment and other randomness (but NOT policy RNG -
              policy RNG must be set during policy creation via create_policy's rng_seed parameter).

    Returns:
        success_rate: Success rate across all evaluation episodes.
    """
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    LIBERO_ENV_RESOLUTION = 256
    RESIZE_SIZE = 224
    REPLAN_STEPS = 5
    VIDEO_OUT_PATH = video_out_path

    # Seed all random number generators for reproducibility
    # Note: Policy RNG must be set during policy creation via create_policy(rng_seed=seed)
    np.random.seed(seed)
    random.seed(seed)

    # Start evaluation
    task_episodes, task_successes = 0, 0
    print(f"\nStarting evaluation: {num_trials} trials (seed={seed})...")

    # Set the JAX compilation cache directory to avoid recompilation and speed up repeated runs.
    jax.config.update(
        "jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser())
    )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    if task_id >= num_tasks_in_suite:
        raise ValueError(
            f"Task ID {task_id} is out of range. Task suite has {num_tasks_in_suite} tasks."
        )

    print(f"Task suite: {task_suite_name}")
    print(f"Evaluating task {task_id} of {num_tasks_in_suite}")

    # Determine max steps
    if task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {task_suite_name}")

    if save_video:
        pathlib.Path(VIDEO_OUT_PATH).mkdir(parents=True, exist_ok=True)

    # Get task
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    # Initialize environment
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
    print(f"Task: {task_description}")

    # Run evaluation episodes
    for episode_idx in tqdm(range(num_trials), desc=f"Task {task_id}"):
        print(f"Episode {episode_idx+1} of {num_trials}")

        # Reset environment
        env.reset()
        env.seed(seed + episode_idx)  # Use episode_idx to ensure different but deterministic seeds per episode
        action_plan = collections.deque()

        # Set initial states
        obs = env.set_init_state(initial_states[episode_idx])

        # Setup
        t = 0
        replay_images = []

        while t < max_steps + num_steps_wait:
            try:
                # Wait for objects to stabilize
                if t < num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Get preprocessed images (rotate 180 degrees to match train preprocessing)
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, RESIZE_SIZE, RESIZE_SIZE)
                )

                # Save for replay video
                replay_images.append(img)

                if not action_plan:
                    # Prepare observations dict
                    curr_obs_dict = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(task_description),
                    }

                    # Query model directly (no websocket)
                    result = policy.infer(curr_obs_dict)
                    action_chunk = result["actions"]
                    assert (
                        len(action_chunk) >= REPLAN_STEPS
                    ), f"Policy only predicts {len(action_chunk)} steps, need {REPLAN_STEPS}"
                    action_plan.extend(action_chunk[:REPLAN_STEPS])

                action = action_plan.popleft()

                # Execute action
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    break
                t += 1

            except Exception as e:
                print(f"Error in episode {episode_idx+1}: {e}")
                break

        task_episodes += 1

        # Save replay video
        suffix = "success" if done else "failure"
        task_segment = task_description.replace(" ", "_")
        if save_video:
            video_filename = (
                f"rollout_task{task_id}_{task_segment}_ep{episode_idx+1}_{suffix}.mp4"
            )
            imageio.mimwrite(
                pathlib.Path(VIDEO_OUT_PATH) / video_filename,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

        # Log progress
        if (episode_idx + 1) % 1 == 0:
            print(
                f"  Episodes: {task_episodes}, Successes: {task_successes} ({task_successes/task_episodes*100:.1f}%)"
            )

    # Final results
    success_rate = task_successes / task_episodes if task_episodes > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"Final Results for Task {task_id}:")
    print(f"  Task: {task_description}")
    print(f"  Episodes: {task_episodes}")
    print(f"  Successes: {task_successes}")
    print(f"  Success rate: {success_rate*100:.1f}%")
    print(f"{'='*60}")

    return success_rate


def make_observation_from_simulator(
    policy: _policy.Policy, curr_obs_dict: dict
) -> _model.Observation:
    # Create Observation object from the current obs_dict
    inputs = jax.tree.map(lambda x: x, curr_obs_dict)
    inputs = policy._input_transform(inputs)
    if not policy._is_pytorch_model:
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        policy._rng, sample_rng_or_pytorch_device = jax.random.split(policy._rng)
    else:
        # Convert inputs to PyTorch tensors and move to correct device
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(policy._pytorch_device)[
                None, ...
            ],
            inputs,
        )
        sample_rng_or_pytorch_device = policy._pytorch_device

    observation = _model.Observation.from_dict(inputs)
    return observation


# Create a simple dataloader that returns the same batch every time
class SingleBatchDataLoader:
    """A simple dataloader that returns the same batch (obs, actions) every time."""

    def __init__(
        self,
        obs: _model.Observation,
        actions: _model.Actions,
        data_config: _config.DataConfig,
    ):
        self.obs = obs
        self.actions = actions
        self._data_config = data_config

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        """Yield the same batch repeatedly."""
        while True:
            yield (self.obs, self.actions)


def copy_model(model, train_config: _config.TrainConfig):
    original_model_graphdef = nnx.graphdef(model)
    original_model_params = nnx.state(model)
    original_model_params = nnx_utils.state_map(
        original_model_params,
        train_config.freeze_filter,
        lambda p: (
            p.replace(p.value.astype(jnp.bfloat16))
            if train_config.freeze_filter is not None
            else p
        ),
    )

    # Create a fresh copy of the original model for training
    def copy_param(x):
        if isinstance(x, jax.Array):
            jax.block_until_ready(x)
            return jnp.array(np.asarray(x))  # Force copy through CPU
        return x

    original_params_copy = jax.tree.map(copy_param, original_model_params)
    jax.block_until_ready(original_params_copy)
    model_copy = nnx.merge(original_model_graphdef, original_params_copy)
    model_copy.eval()
    # NOTE: No need to block here - nnx.merge() doesn't create new arrays, just combines already-materialized params
    return model_copy


def run_evaluation_ttt(
    policy: _policy.Policy,
    nn_fetcher: Any,
    train_config: _config.TrainConfig,
    dataset: Any,
    num_trials: int = 10,
    task_suite_name: str = "libero_90",
    task_id: int = 0,
    num_steps_wait: int = 10,
    save_video: bool = True,
    video_out_path: str = "data/libero/videos",
    seed: int = 0,
    ttt_num_steps: int = 10,
    ttt_frequency: int = 50,
    learning_rate: float = 2.5e-5,
    ttt_k: int = 50,
    ttt_use_modalities: Optional[List[str]] = None,
    plot_observations: bool = False,
):
    """
    Run evaluation with test-time training (TTT).

    Args:
        policy: The policy to evaluate
        nn_fetcher: NearestNeighborFetcher object for retrieving similar samples
        train_config: Training configuration for fine-tuning
        dataset: The dataset or dataloader to index (will extract dataset if dataloader is passed)
        num_trials: Number of evaluation episodes
        task_suite_name: Name of the LIBERO task suite
        task_id: ID of the task to evaluate
        num_steps_wait: Number of steps to wait for environment stabilization
        save_video: Whether to save rollout videos
        video_out_path: Path to save videos
        seed: Random seed for reproducibility
        ttt_num_steps: Number of gradient steps for each TTT update
        ttt_frequency: Perform TTT every N steps during rollout
        learning_rate: Learning rate for TTT fine-tuning
        warmup_steps: Number of warmup steps for TTT optimizer
        ttt_k: Number of nearest neighbors to retrieve for TTT
        ttt_batch_size: Batch size for TTT training
        ttt_use_modalities: List of modalities to use for retrieval (default: all available)

    Returns:
        success_rate: Success rate across all evaluation episodes
    """
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    LIBERO_ENV_RESOLUTION = 256
    RESIZE_SIZE = 224
    REPLAN_STEPS = 5
    NUM_STEPS_WAIT = 10
    VIDEO_OUT_PATH = video_out_path
    CHECKPOINT_DIR = (
        "/cluster/home/anmari/.cache/openpi/openpi-assets/checkpoints/pi05_libero"
    )

    # Seed all random number generators for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    jax_key = jax.random.PRNGKey(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    # Start evaluation
    task_episodes, task_successes = 0, 0
    # Collect per-episode metrics for external logging / plotting (e.g. ttt_evaluation.py).
    # We will attach this data to the function object at the end without changing
    # the public return type.
    all_episode_metrics: list[dict[str, Any]] = []
    print(f"\nStarting TTT evaluation: {num_trials} trials (seed={seed})...")
    print(f"TTT settings: {ttt_num_steps} steps every {ttt_frequency} rollout steps")

    # Set the JAX compilation cache directory to avoid recompilation and speed up repeated runs.
    jax.config.update(
        "jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser())
    )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    if task_id >= num_tasks_in_suite:
        raise ValueError(
            f"Task ID {task_id} is out of range. Task suite has {num_tasks_in_suite} tasks."
        )

    print(f"Task suite: {task_suite_name}")
    print(f"Evaluating task {task_id} of {num_tasks_in_suite}")

    # Determine max steps
    if task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {task_suite_name}")

    if save_video:
        pathlib.Path(VIDEO_OUT_PATH).mkdir(parents=True, exist_ok=True)

    # Get task
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    # Initialize environment
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
    print(f"Task: {task_description}")

    # Initialize train state for TTT (we'll update the policy's model during TTT)
    train_state = None
    ttt_count = 0
    original_model = policy._model

    print_trainable_parameters(nnx.state(original_model), train_config.trainable_filter)

    # Run evaluation episodes
    pbar = tqdm(range(num_trials), desc=f"Task {task_id} | Success: 0/0 (0.0%)")
    for episode_idx in pbar:
        pbar.write(f"Episode {episode_idx+1} of {num_trials}")
        # Reset environment
        env.seed(seed + episode_idx)  # Use episode_idx to ensure different but deterministic seeds per episode
        env.reset()

        action_plan = collections.deque()
        # Set initial states
        obs = env.set_init_state(initial_states[episode_idx])

        # check image determinism
        raw_img_sum = float(np.sum(obs["agentview_image"]))
        raw_wrist_img_sum = float(np.sum(obs["robot0_eye_in_hand_image"]))
        raw_img_mean = float(np.mean(obs["agentview_image"]))
        raw_wrist_img_mean = float(np.mean(obs["robot0_eye_in_hand_image"]))
        pbar.write(
            f"[ENV CHECK] Raw env images START - base: sum={raw_img_sum:.10f}, mean={raw_img_mean:.10f}, "
            f"wrist: sum={raw_wrist_img_sum:.10f}, mean={raw_wrist_img_mean:.10f}"
        )
        # raise ValueError("Stop here")

        # Setup
        t = 0
        replay_images = []
        distances_actions = []
        similarities = []
        episode_losses: list[list[float]] = []

        while t < max_steps + num_steps_wait:
            # Wait for objects to stabilize
            if t < num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            # Get preprocessed images (rotate 180 degrees to match train preprocessing)
            # Check raw images from environment at specific steps (0, 1, 5, 10) and TTT steps to trace non-determinism
            check_determinism = False
            is_ttt_step = (t - num_steps_wait) % ttt_frequency == 0
            check_steps = {0, 1, 5, 10}
            should_check_images = check_determinism and (is_ttt_step or t in check_steps)

            if should_check_images:
                raw_img_sum = float(np.sum(obs["agentview_image"]))
                raw_wrist_img_sum = float(np.sum(obs["robot0_eye_in_hand_image"]))
                raw_img_mean = float(np.mean(obs["agentview_image"]))
                raw_wrist_img_mean = float(np.mean(obs["robot0_eye_in_hand_image"]))
                pbar.write(
                    f"[ENV CHECK] Raw env images (t={t}) - base: sum={raw_img_sum:.10f}, mean={raw_img_mean:.10f}, "
                    f"wrist: sum={raw_wrist_img_sum:.10f}, mean={raw_wrist_img_mean:.10f}"
                )

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(
                obs["robot0_eye_in_hand_image"][::-1, ::-1]
            )

            # Check after rotation
            if should_check_images:
                rotated_img_sum = float(np.sum(img))
                rotated_wrist_img_sum = float(np.sum(wrist_img))
                pbar.write(
                    f"[ENV CHECK] After rotation (t={t}) - base: sum={rotated_img_sum:.10f}, "
                    f"wrist: sum={rotated_wrist_img_sum:.10f}"
                )

            img_resized = image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE)
            wrist_img_resized = image_tools.resize_with_pad(wrist_img, RESIZE_SIZE, RESIZE_SIZE)

            # Check after resize
            if should_check_images:
                resized_img_sum = float(np.sum(img_resized))
                resized_wrist_img_sum = float(np.sum(wrist_img_resized))
                pbar.write(
                    f"[ENV CHECK] After resize (t={t}) - base: sum={resized_img_sum:.10f}, "
                    f"wrist: sum={resized_wrist_img_sum:.10f}"
                )

            img = image_tools.convert_to_uint8(img_resized)
            wrist_img = image_tools.convert_to_uint8(wrist_img_resized)

            # Check after uint8 conversion
            if should_check_images:
                uint8_img_sum = float(np.sum(img))
                uint8_wrist_img_sum = float(np.sum(wrist_img))
                uint8_img_mean = float(np.mean(img))
                uint8_wrist_img_mean = float(np.mean(wrist_img))
                pbar.write(
                    f"[ENV CHECK] After uint8 (t={t}) - base: sum={uint8_img_sum:.10f}, mean={uint8_img_mean:.10f}, "
                    f"wrist: sum={uint8_wrist_img_sum:.10f}, mean={uint8_wrist_img_mean:.10f}"
                )

            # Save for replay video
            replay_images.append(img)

            if not action_plan:
                # Prepare observations dict
                curr_obs_dict = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }

                # Query model directly (no websocket)
                action_chunk = policy.infer(curr_obs_dict)["actions"]

                # Check action_chunk before TTT for determinism
                if check_determinism and is_ttt_step:
                    action_chunk_jax_check = (
                        action_chunk
                        if isinstance(action_chunk, jax.Array)
                        else jnp.array(action_chunk)
                    )
                    action_chunk_before_ttt_sum = float(jax.device_get(jnp.sum(action_chunk_jax_check)))
                    action_chunk_before_ttt_mean = float(jax.device_get(jnp.mean(action_chunk_jax_check)))
                    action_chunk_before_ttt_norm = float(jax.device_get(jnp.linalg.norm(action_chunk_jax_check)))
                    pbar.write(
                        f"[TTT] Action chunk BEFORE TTT (t={t}) - sum: {action_chunk_before_ttt_sum:.10f}, "
                        f"mean: {action_chunk_before_ttt_mean:.10f}, norm: {action_chunk_before_ttt_norm:.10f}"
                    )

                assert (
                    len(action_chunk) >= REPLAN_STEPS
                ), f"Policy only predicts {len(action_chunk)} steps, need {REPLAN_STEPS}"

                # Perform TTT if at the right frequency
                if (t - num_steps_wait) % ttt_frequency == 0:
                    ttt_count += 1
                    pbar.write(f"[TTT {ttt_count}] Starting TTT update")

                    # Check for non-determinism: generate random numbers from all sources
                    # This helps identify if randomness is consistent across runs with the same seed
                    check_determinism = True
                    if check_determinism:
                        jax_key, jax_subkey = jax.random.split(jax_key)
                        jax_random_val = float(jax.device_get(jax.random.uniform(jax_subkey)))
                        python_random_val = random.random()
                        numpy_random_val = np.random.rand()
                        pbar.write(
                            f"[TTT {ttt_count}] Determinism check - JAX: {jax_random_val:.10f}, "
                            f"Python: {python_random_val:.10f}, NumPy: {numpy_random_val:.10f}, "
                            f"t={t}, episode={episode_idx+1}"
                        )

                    # NOTE: Don't clear caches here - it forces recompilation of train_step_jit
                    # jax.clear_caches()  # Disabled to avoid recompilation overhead

                    # Check time of fetching neighbors
                    start_time = time.time()
                    # Create Observation object from the current observation
                    observation = make_observation_from_simulator(
                        policy, curr_obs_dict
                    )

                    # Check observation hash for determinism (before fetching neighbors)
                    if check_determinism:

                        # 'base_0_rgb' and 'left_wrist_0_rgb'

                        # Compute a hash of the observation to check if it's the same across runs
                        obs_image_sum = float(jax.device_get(jnp.sum(observation.images["base_0_rgb"])))
                        obs_image2_sum = float(jax.device_get(jnp.sum(observation.images["left_wrist_0_rgb"])))
                        obs_state_sum = float(jax.device_get(jnp.sum(observation.state)))
                        pbar.write(
                            f"[TTT {ttt_count}] Observation checksum - image1_sum: {obs_image_sum:.6f}, "
                            f"image2_sum: {obs_image2_sum:.6f}, state_sum: {obs_state_sum:.6f}"
                        )

                    # Fetch nearest neighbor inxdices based on current observation
                    distances, indices, metadata = nn_fetcher.fetch_neighbors(
                        observation=observation,
                        use_modalities=ttt_use_modalities,
                        k=ttt_k,
                    )
                    end_time = time.time()
                    pbar.write(
                        f"[TTT {ttt_count}] ({end_time - start_time:.2f}s) Retrieved neighbors with similarities: {distances[:min(5, len(distances))]}"
                    )

                    # Check if fetched indices are deterministic
                    if check_determinism:
                        indices_str = str(list(indices[:min(10, len(indices))]))
                        distances_str = str([float(d) for d in distances[:min(5, len(distances))]])
                        # Check raw distance values with high precision to detect floating-point differences
                        distances_precise = [f"{float(d):.15f}" for d in distances[:min(5, len(distances))]]
                        pbar.write(
                            f"[TTT {ttt_count}] Fetched indices (first 10): {indices_str}, "
                            f"distances (first 5): {distances_str}, "
                            f"distances (precise): {distances_precise}"
                        )

                    obs, actions_fetched = fetch_samples(dataset, indices)
                    fetched_observation = obs

                    # Check fetched data for determinism
                    if check_determinism:
                        # Check a sample of the fetched actions
                        if isinstance(actions_fetched, jax.Array):
                            actions_sum = float(jax.device_get(jnp.sum(actions_fetched)))
                            actions_mean = float(jax.device_get(jnp.mean(actions_fetched)))
                        else:
                            actions_sum = float(np.sum(actions_fetched))
                            actions_mean = float(np.mean(actions_fetched))
                        pbar.write(
                            f"[TTT {ttt_count}] Fetched actions checksum - sum: {actions_sum:.6f}, mean: {actions_mean:.6f}"
                        )

                    # Plot current observation
                    if plot_observations:
                        plot_observation_with_decoded_prompt(
                            observation=observation,
                            similarity_score=None,
                            plot_title_prefix=f"[TTT {ttt_count}] Current Observation",
                        )

                        # Plot fetched observation (first/best neighbor)
                        # Note: obs might be a batch, but we'll plot the first one which is the best match
                        if len(indices) > 0 and len(distances) > 0:
                            # Extract first observation from batch if needed
                            # If obs is a batch, plot_observation_images should handle extracting the first element
                            plot_observation_with_decoded_prompt(
                                observation=fetched_observation,
                                similarity_score=distances[0],
                                plot_title_prefix=f"[TTT {ttt_count}] Fetched Observation (Best Match)",
                            )

                    # Prepare dataloader for TTT
                    start_time = time.time()
                    ttt_data_config = train_config.data.create(
                        train_config.assets_dirs, train_config.model
                    )
                    ttt_data_loader = SingleBatchDataLoader(
                        obs, actions_fetched, ttt_data_config
                    )
                    end_time = time.time()
                    pbar.write(
                        f"[TTT {ttt_count}] ({end_time - start_time:.2f}s) Prepared dataloader"
                    )

                    # Debug: Check if train_config object is the same between TTT calls

                    # Restore original model before TTT so each TTT starts from the same baseline
                    print_memory_checkpoint(
                        f"[TTT {ttt_count}] Before copying original model", 1288
                    )
                    time_start = time.time()
                    model_copy = copy_model(policy._model, train_config)
                    time_end = time.time()
                    pbar.write(
                        f"[TTT {ttt_count}] ({time_end - time_start:.2f}s) Copied original model"
                    )
                    print_memory_checkpoint(
                        f"[TTT {ttt_count}] After copying original model", 1290
                    )
                    # NOTE: here we go 6.25 GB -> 12.58 GB as the model is copied

                    # Check model parameters before TTT for determinism
                    if check_determinism:
                        model_params = nnx.state(model_copy)
                        # Get a sample parameter to check
                        sample_param = next(iter(jax.tree.leaves(model_params.filter(nnx.Param))))
                        if hasattr(sample_param, "value"):
                            param_val = float(jax.device_get(sample_param.value.flatten()[0]))
                            param_norm = float(jax.device_get(jnp.linalg.norm(sample_param.value)))
                        else:
                            param_val = float(jax.device_get(sample_param.flatten()[0]))
                            param_norm = float(jax.device_get(jnp.linalg.norm(sample_param)))
                        pbar.write(
                            f"[TTT {ttt_count}] Model param before TTT - first_val: {param_val:.10f}, norm: {param_norm:.6f}"
                        )

                    # Perform fine-tuning on the copy (with donation enabled to save memory)
                    # Each TTT should start fresh - don't resume from previous TTT state to avoid memory accumulation
                    # Use donation=True to save memory - the copy will be modified, original is preserved
                    trained_model, losses, train_state = train_model_on_fly(
                        model=model_copy,  # Pass copy, original model is preserved
                        training_data_loader=ttt_data_loader,
                        config=train_config,
                        learning_rate=learning_rate,
                        num_steps=ttt_num_steps,
                        warmup_steps=0,
                        weight_decay=0.0,
                        log_interval=max(1, ttt_num_steps // 2),
                        seed=seed + ttt_count,  # Different seed for each TTT
                        resume_train_state=None,  # Each TTT starts fresh - don't accumulate optimizer state
                        resume_losses=None,  # Don't carry over losses
                        donate_buffers=True,  # Enable donation to save memory (copy is modified, original preserved)
                    )
                    # Print losses
                    # Store a copy of the losses for this TTT update so external
                    # consumers can visualize them later.
                    episode_losses.append(list(losses))

                    # Check first and last loss for determinism
                    if check_determinism and len(losses) > 0:
                        first_loss = losses[0]
                        last_loss = losses[-1]
                        pbar.write(
                            f"[TTT {ttt_count}] Loss values - first: {first_loss:.10f}, last: {last_loss:.10f}, "
                            f"all: {[f'{l:.6f}' for l in losses]}"
                        )
                    # Use fine-tuned model to generate new action plan
                    # NOTE: No need to block - trained_model was already returned from train_model_on_fly
                    # which handled all necessary blocking during the copy/merge process
                    trained_model.eval()

                    # Update the policy's model temporarily
                    # Use deterministic seed for TTT policy to ensure reproducibility
                    policy._model = trained_model
                    #ttt_policy = create_policy(
                    #    trained_model, train_config, CHECKPOINT_DIR,
                    #    rng_seed=seed + ttt_count  # Deterministic seed for each TTT
                    #)
                    # Generate new action_chunk with fine-tuned model
                    # ttt_action_chunk = ttt_policy.infer(curr_obs_dict)["actions"]
                    ttt_action_chunk = policy.infer(curr_obs_dict)["actions"]

                    # Compare action_chunk with ttt_action_chunk
                    # Compute distance on GPU using JAX to avoid expensive GPU->CPU transfer
                    # Only transfer the final scalar result, not the full arrays
                    action_chunk_jax = (
                        action_chunk
                        if isinstance(action_chunk, jax.Array)
                        else jnp.array(action_chunk)
                    )
                    ttt_action_chunk_jax = (
                        ttt_action_chunk
                        if isinstance(ttt_action_chunk, jax.Array)
                        else jnp.array(ttt_action_chunk)
                    )

                    # Check action chunks for determinism before computing distance
                    if check_determinism:
                        action_chunk_sum = float(jax.device_get(jnp.sum(action_chunk_jax)))
                        action_chunk_mean = float(jax.device_get(jnp.mean(action_chunk_jax)))
                        action_chunk_norm = float(jax.device_get(jnp.linalg.norm(action_chunk_jax)))
                        ttt_action_chunk_sum = float(jax.device_get(jnp.sum(ttt_action_chunk_jax)))
                        ttt_action_chunk_mean = float(jax.device_get(jnp.mean(ttt_action_chunk_jax)))
                        ttt_action_chunk_norm = float(jax.device_get(jnp.linalg.norm(ttt_action_chunk_jax)))
                        diff_sum = float(jax.device_get(jnp.sum(action_chunk_jax - ttt_action_chunk_jax)))
                        pbar.write(
                            f"[TTT {ttt_count}] Action chunks checksum - "
                            f"action_chunk: sum={action_chunk_sum:.10f}, mean={action_chunk_mean:.10f}, norm={action_chunk_norm:.10f}, "
                            f"ttt_action_chunk: sum={ttt_action_chunk_sum:.10f}, mean={ttt_action_chunk_mean:.10f}, norm={ttt_action_chunk_norm:.10f}, "
                            f"diff_sum={diff_sum:.10f}"
                        )

                    # Compute distance on GPU
                    action_distance_jax = jnp.linalg.norm(
                        action_chunk_jax - ttt_action_chunk_jax
                    )
                    # Only transfer the scalar result (much faster than transferring full arrays)
                    action_distance = float(jax.device_get(action_distance_jax))
                    distances_actions.append((t, action_distance))
                    similarities.append((t, distances[0]))
                    if check_determinism:
                        pbar.write(
                            f"[TTT {ttt_count}] Distance between action_chunk and actions_fetched: {action_distance:.10f} (precise)"
                        )
                    else:
                        pbar.write(
                            f"[TTT {ttt_count}] Distance between action_chunk and actions_fetched: {action_distance:.4f}"
                        )

                    # Explicitly delete all TTT-related objects to free memory
                    # NOTE: check if needed the next line
                    action_chunk = ttt_action_chunk
                    policy._model = original_model
                    del trained_model, model_copy, train_state, losses

                    # Force garbage collection and clear JAX caches
                    # import gc
                    # gc.collect()
                    # jax.clear_caches()

                action_plan.extend(action_chunk[:REPLAN_STEPS])

            action = action_plan.popleft()

            # Execute action
            obs, reward, done, info = env.step(action.tolist())
            if done:
                task_successes += 1
                break
            t += 1


        task_episodes += 1

        # Update progress bar description with current success rate
        success_rate = (task_successes / task_episodes * 100) if task_episodes > 0 else 0.0
        pbar.set_description(f"Task {task_id} | Success: {task_successes}/{task_episodes} ({success_rate:.1f}%)")

        # Save episode-level metrics for external plotting instead of showing inline.
        all_episode_metrics.append(
            {
                "episode_idx": episode_idx,
                "success": bool(done),
                "distances_actions": list(distances_actions),
                "similarities": list(similarities),
                "losses": episode_losses,
            }
        )

        # Save replay video
        suffix = "success" if done else "failure"
        task_segment = task_description.replace(" ", "_")
        if save_video:
            video_filename = f"rollout_ttt_task{task_id}_{task_segment}_ep{episode_idx+1}_{suffix}.mp4"
            imageio.mimwrite(
                pathlib.Path(VIDEO_OUT_PATH) / video_filename,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

        # Log progress
        if (episode_idx + 1) % 1 == 0:
            pbar.write(
                f"  Episodes: {task_episodes}, Successes: {task_successes} ({task_successes/task_episodes*100:.1f}%)"
            )

    # Expose collected metrics on the function object without changing the
    # public return type, so scripts can read them after a call.
    try:
        run_evaluation_ttt.last_episode_metrics = all_episode_metrics  # type: ignore[attr-defined]
    except Exception:
        pass

    # Final results
    success_rate = task_successes / task_episodes if task_episodes > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"Final TTT Results for Task {task_id}:")
    print(f"  Task: {task_description}")
    print(f"  Episodes: {task_episodes}")
    print(f"  Successes: {task_successes}")
    print(f"  Success rate: {success_rate*100:.1f}%")
    print(f"  Total TTT updates: {ttt_count}")
    print(f"{'='*60}")

    return success_rate


def create_policy(
    model: _model.BaseModel,
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    rng_seed: int | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        rng_seed: Random seed for JAX RNG key. If provided, ensures reproducible policy inference.
                  If None, defaults to 0.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    # TODO: check how to provide data_config here
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    # TODO: check how to provide norm_stats for TTT
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(
            pathlib.Path(checkpoint_dir) / "assets", data_config.asset_id
        )

    # Create RNG key if seed provided for reproducible sampling
    rng_key = jax.random.PRNGKey(rng_seed) if rng_seed is not None else None

    return _policy.Policy(
        model,
        # rng=rng_key, NOTE: disabled, this is bugged in OpenPI
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=False,
        pytorch_device=None,
    )
