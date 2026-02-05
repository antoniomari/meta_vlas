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
    repeat_batch: int = 1,
    reset_policy: bool = True,
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
    ttt_data_config = train_config.data.create(
        train_config.assets_dirs, train_config.model
    )

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
        # Reset policy model to original model at the beginning of each episode
        policy._model = original_model

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

            # Resize and convert image
            img_resized = image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE)
            img = image_tools.convert_to_uint8(img_resized)
            # Resize and convert wrist image
            wrist_img_resized = image_tools.resize_with_pad(wrist_img, RESIZE_SIZE, RESIZE_SIZE)
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

            # Save for replay video (with step number overlaid on frame)
            replay_images.append(_draw_step_on_frame(img, t))

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

                    obs, actions_fetched = fetch_samples(dataset, indices, repeat=repeat_batch)
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
                                similarity_score=distances,
                                plot_title_prefix=f"[TTT {ttt_count}] Fetched Observation (Best Match)",
                            )

                    # Prepare dataloader for TTT
                    start_time = time.time()

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
                    ttt_policy = new_policy_like(policy, trained_model)
                    #ttt_policy = create_policy(
                    #    trained_model, train_config, CHECKPOINT_DIR,
                    #    rng_seed=seed + ttt_count  # Deterministic seed for each TTT
                    #)
                    # Generate new action_chunk with fine-tuned model
                    # ttt_action_chunk = ttt_policy.infer(curr_obs_dict)["actions"]
                    ttt_action_chunk = ttt_policy.infer(curr_obs_dict)["actions"]

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

                    # Optionally plot current loss and action distances at end of each TTT step
                    if plot_observations:
                        fig, (ax_loss, ax_dist) = plt.subplots(1, 2, figsize=(10, 4))
                        if losses:
                            ax_loss.plot(range(len(losses)), losses, "b-o", markersize=3)
                        ax_loss.set_title(f"[TTT {ttt_count}] Loss (this update)")
                        ax_loss.set_xlabel("Gradient step")
                        ax_loss.set_ylabel("Loss")
                        if distances_actions:
                            steps_ttt = [x[0] for x in distances_actions]
                            dists = [x[1] for x in distances_actions]
                            ax_dist.plot(steps_ttt, dists, "g-o", markersize=3)
                        ax_dist.set_title(f"[TTT {ttt_count}] Action distance vs env step")
                        ax_dist.set_xlabel("Env step t")
                        ax_dist.set_ylabel("Action distance")
                        plt.tight_layout()
                        plt.show()
                        plt.close(fig)

                    # Explicitly delete all TTT-related objects to free memory
                    # NOTE: check if needed the next line
                    action_chunk = ttt_action_chunk

                    if reset_policy:
                        del trained_model
                    else:
                        # NOTE: for now, it will not restore the right original model at the beginning of the next episode
                        # The original model now becomes the TTT model
                        del original_model
                        original_model = trained_model
                        new_policy = new_policy_like(policy, original_model)
                        del policy
                        policy = new_policy

                    del model_copy, train_state, losses, ttt_data_loader, obs, actions_fetched
                    print_memory_checkpoint(f"[TTT {ttt_count}] At the end of the TTT update deleting objects", 1300)

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

        # Save per-episode losses plot (sequence of losses for each TTT step)
        if episode_losses:
            n_ttt = len(episode_losses)
            n_cols = min(3, n_ttt)
            n_rows = (n_ttt + n_cols - 1) // n_cols
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
            )
            axes_flat = axes.ravel()
            for i, step_losses in enumerate(episode_losses):
                ax = axes_flat[i]
                ax.plot(range(len(step_losses)), step_losses, "b-o", markersize=2)
                ax.set_title(f"TTT update {i+1} ({len(step_losses)} steps)")
                ax.set_xlabel("Gradient step")
                ax.set_ylabel("Loss")
            for j in range(i + 1, len(axes_flat)):
                axes_flat[j].set_visible(False)
            plt.suptitle(f"Episode {episode_idx+1} – {'success' if done else 'failure'}")
            plt.tight_layout()
            losses_plot_path = pathlib.Path(VIDEO_OUT_PATH).parent / f"losses_ep_{episode_idx+1}.pdf"
            plt.savefig(losses_plot_path, bbox_inches="tight")
            plt.close(fig)

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

