## Training Function
import contextlib
import copy
import dataclasses
import functools
from typing import Any, Iterator, SupportsIndex, Tuple
import os
import logging
import etils.epath as epath

# Evaluate pretrained model on first task of libero_90
import collections
import math
import pathlib
import imageio
import sys
import random


from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
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


import torch

# To fix versioning issues with torch
torch.serialization.add_safe_globals(
    [
        np.core.multiarray._reconstruct,  # noqa
        np.ndarray,
        np.dtype,
        np.dtypes.Float64DType,
    ]
)


class CustomDataLoader(_data_loader.DataLoader):
    """Custom data loader for LIBERO fine-tuning that wraps an iterator."""

    def __init__(self, data_config: _config.DataConfig, iterator: Iterator[Tuple[_model.Observation, _model.Actions]]):
        self._data_config = data_config
        self._iterator = iterator

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self) -> Iterator[Tuple[_model.Observation, _model.Actions]]:
        return iter(self._iterator)


def _batch_to_jax(batch):
    """Ensure that each element of the batch (obs, actions) is a JAX array.

    Recursively converts numpy arrays to JAX arrays while preserving the structure
    of Observation and Actions objects.

    Note: Arrays are kept on CPU initially. JAX will move them to GPU during JIT
    compilation, ensuring only one batch is on GPU at a time.
    """
    if isinstance(batch, tuple) and len(batch) == 2:
        obs, actions = batch

        # Recursively convert numpy arrays to JAX arrays using tree_map
        # This preserves the structure of Observation (nested dicts) and Actions
        # Use jnp.asarray instead of device_put to keep on CPU until JIT moves it
        def _to_jax_array(x):
            if isinstance(x, np.ndarray):
                # Create a fresh JAX array from numpy array
                return jnp.asarray(x)  # Keep on CPU, JIT will move to GPU
            elif isinstance(x, jax.Array):
                # For JAX arrays, check if deleted and create a new one if needed
                try:
                    # Try to access the array to see if it's deleted
                    _ = x.shape  # This will raise if deleted
                    # If not deleted, return as-is (JIT will handle placement)
                    return x
                except RuntimeError:
                    # If deleted, we can't recover - this shouldn't happen with fresh batches
                    raise RuntimeError("Attempted to use a deleted JAX array in batch")
            else:
                return x  # Preserve non-array types (e.g., None, bool, etc.)

        obs = jax.tree.map(_to_jax_array, obs)
        actions = jax.tree.map(_to_jax_array, actions)
        return (obs, actions)
    else:
        raise ValueError("Batch must be a Tuple of (observation, actions)")


def train_model_on_fly(
    model: _model.BaseModel,
    training_data_loader: _data_loader.DataLoader,
    learning_rate: float = 2.5e-5,
    num_steps: int = 1000,
    warmup_steps: int = 100,
    weight_decay: float = 0.0,
    trainable_filter: Any = None,
    freeze_filter: Any = None,
    log_interval: int = 100,
    seed: int = 42,
    # Resume parameters - pass these to continue training from a previous run
    resume_train_state: training_utils.TrainState | None = None,
    resume_losses: list[float] | None = None,
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
    # Suppress JAX compilation warnings for cleaner output in notebooks
    logging.getLogger('absl').setLevel(logging.ERROR)  # Suppress JAX/absl warnings

    # Seed all random number generators for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Set the JAX compilation cache directory to avoid recompilation and speed up repeated runs.
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # Speed up JIT compilation by reducing XLA autotuning (faster compilation, minimal performance impact)
    if 'XLA_FLAGS' not in os.environ:
        os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
    elif '--xla_gpu_autotune_level' not in os.environ.get('XLA_FLAGS', ''):
        os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_autotune_level=0'

    # Determine trainable filter
    if trainable_filter is not None:
        _trainable_filter_for_init = trainable_filter
    elif freeze_filter is not None:
        _trainable_filter_for_init = nnx.All(nnx.Param, nnx.Not(freeze_filter))
    else:
        _trainable_filter_for_init = nnx.Param

    # Check if resuming from a previous train state
    if resume_train_state is not None:
        # Use the provided train state directly
        train_state = resume_train_state
        start_step = int(train_state.step)
        end_step = start_step + num_steps
        print(f"Resuming training from step {start_step}, will train {num_steps} more steps to step {end_step}")

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

        # Create a deep copy of the model to avoid modifying the original
        graphdef, state = nnx.split(model)
        model_copy = nnx.merge(graphdef, state)

        # Initialize optimizer with cosine decay schedule
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate / (warmup_steps + 1),
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=num_steps,
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

        # Get model parameters
        params = nnx.state(model_copy)

        # Apply freeze filter if provided
        if freeze_filter is not None:
            params = nnx_utils.state_map(
                params, freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
            )

        # Filter trainable parameters
        trainable_params = params.filter(_trainable_filter_for_init)

        # Count trainable parameters
        def param_count(x):
            if hasattr(x, "value"):
                return int(jax.device_get(jnp.size(x.value)))
            return int(jax.device_get(jnp.size(x)))
        param_counts = jax.tree.map(param_count, trainable_params)
        num_trainable_params = sum(jax.tree_util.tree_leaves(param_counts))
        del param_counts
        print(f"Number of trainable parameters: {num_trainable_params:,}")

        # Initialize optimizer state
        opt_state = tx.init(trainable_params)
        del trainable_params

        # Get graphdef for the model (static structure)
        graphdef = nnx.graphdef(model_copy)

        # Create TrainState
        train_state = training_utils.TrainState(
            step=0,
            params=params,
            model_def=graphdef,
            tx=tx,
            opt_state=opt_state,
            ema_decay=None,
            ema_params=None,
        )

    # Initialize RNG
    rng = jax.random.key(seed)

    # Define trainable_filter for training step
    _trainable_filter = _trainable_filter_for_init

    # Training step function - mirrors train_step from scripts/train.py
    def train_step(rng, state, batch):
        model = nnx.merge(state.model_def, state.params)
        model.train()

        # Note: @at.typecheck removed for faster compilation (it's expensive during JIT)
        def loss_fn(
            model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
        ):
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            return jnp.mean(chunked_loss)

        train_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch

        # Filter out frozen params - use trainable_filter from closure
        diff_state = nnx.DiffState(0, _trainable_filter)
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

        params = state.params.filter(_trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Update the model in place and return the new full state
        # new_params is a filtered State (only trainable params), nnx.update will update only those
        nnx.update(model, new_params)
        # Get the full updated state from the model
        new_params = nnx.state(model)

        new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)

        # Convert grads (NNX State) to pure dict for optax.global_norm
        # grads from nnx.value_and_grad returns a State, need to extract .value from Params
        # Use to_pure_dict if available, otherwise extract values
        if hasattr(grads, 'to_pure_dict'):
            grads_dict = grads.to_pure_dict()
        else:
            # Fallback: extract .value from Param objects
            def get_grad_value(x):
                if hasattr(x, 'value'):
                    return x.value
                return x
            grads_dict = jax.tree.map(get_grad_value, grads)

        grad_norm = optax.global_norm(grads_dict)

        info = {
            "loss": loss,
            "grad_norm": grad_norm,
        }
        return new_state, info

    # JIT compile the training step with optimizations
    # donate_argnums=(1,) donates the train_state buffer to avoid copying (like train.py)
    # Note: After donation, the original train_state should not be accessed
    train_step_jit = jax.jit(train_step, donate_argnums=(1,))

    # Training loop
    # Initialize losses from resume_losses if provided, otherwise start fresh
    losses = list(resume_losses) if resume_losses else []
    infos = []

    # Use tqdm.auto for notebook compatibility - it automatically detects notebook environment
    # Train from start_step to end_step (num_steps additional steps)
    pbar = tqdm(
        range(start_step, end_step),
        desc="Training",
        total=num_steps,  # Show progress as num_steps (the additional steps)
        dynamic_ncols=True,
        mininterval=0.5,
        maxinterval=2.0
    )

    # Initialize iterator
    data_iter = iter(training_data_loader)
    # Warm-up compilation: compile with first batch before starting training loop
    print("Compiling training step (this may take a few minutes)...")
    try:
        warmup_batch = next(data_iter)
        warmup_batch = _batch_to_jax(warmup_batch)
        # Ensure batch is fully materialized before JIT call
        # jax.tree.map(lambda x: jax.block_until_ready(x) if isinstance(x, jax.Array) else x, warmup_batch)
        # Trigger compilation - train_state will be donated, so we get a new one back
        new_train_state, warmup_info = train_step_jit(rng, train_state, warmup_batch)
        # Block on just the loss scalar to ensure computation is complete (avoid blocking on anything that might reference donated buffer)
        jax.block_until_ready(warmup_info["loss"])
        train_state = new_train_state  # Update to the new state (old one was donated)
        print("Compilation complete! Starting training...")
    except StopIteration:
        # If iterator is empty, restart it
        if callable(training_set):
            data_iter = training_set()
        else:
            data_iter = iter(training_set)
        warmup_batch = next(data_iter)
        warmup_batch = _batch_to_jax(warmup_batch)
        new_train_state, warmup_info = train_step_jit(rng, train_state, warmup_batch)
        # Block on just the loss scalar to ensure computation is complete (avoid blocking on anything that might reference donated buffer)
        jax.block_until_ready(warmup_info["loss"])
        train_state = new_train_state  # Update to the new state (old one was donated)
        print("Compilation complete! Starting training...")

    for step in pbar:
        # Get batch from training set - aligned with train.py
        try:
            batch = next(data_iter)
        except StopIteration:
            # Restart iterator if exhausted
            if callable(training_set):
                data_iter = training_set()
            else:
                data_iter = iter(training_set)
            batch = next(data_iter)

        # CONVERT batch to jnp arrays
        batch = _batch_to_jax(batch)

        # Training step - aligned with train.py: pass rng, state, batch
        train_state, info = train_step_jit(rng, train_state, batch)

        # Logging - aligned with train.py info structure
        loss_val = float(jax.device_get(info["loss"]))
        grad_norm = float(jax.device_get(info["grad_norm"]))
        losses.append(loss_val)  # Store loss for plotting
        infos.append({"loss": loss_val, "grad_norm": grad_norm})

        # Update progress bar after every step
        pbar.set_postfix({"loss": f"{loss_val:.4f}", "grad_norm": f"{grad_norm:.4f}"})

        # Debug: Check if parameters are actually changing (only on first few steps after start)
        if step < start_step + 3:
            # Get a sample parameter to check if it's updating
            sample_param = next(iter(jax.tree.leaves(train_state.params.filter(_trainable_filter))))
            if hasattr(sample_param, 'value'):
                param_val = float(jax.device_get(sample_param.value.flat[0]))
                print(f"  Debug step {step}: sample param value = {param_val:.6f}, grad_norm = {grad_norm:.6f}")

        if step % log_interval == 0 or step == end_step - 1:
            avg_loss = np.mean([info["loss"] for info in infos[-log_interval:]])
            avg_grad_norm = np.mean([info["grad_norm"] for info in infos[-log_interval:]])
            print(f"Step {step}: loss={avg_loss:.4f}, grad_norm={avg_grad_norm:.4f}")

    print(f"\nTraining completed! Final loss: {losses[-1]:.4f}")

    # Return a copy of the trained model from final train_state
    trained_model = nnx.merge(train_state.model_def, train_state.params)

    # Return model, losses, and train_state (for potential resumption)
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
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
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
    task_description: str = "Task 0",
    seed: int = 0,
):
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    LIBERO_ENV_RESOLUTION = 256
    RESIZE_SIZE = 224
    REPLAN_STEPS = 5
    NUM_STEPS_WAIT = 10
    VIDEO_OUT_PATH = "data/libero/videos"
    CHECKPOINT_CONFIG = "pi05_libero"
    CHECKPOINT_DIR = "gs://openpi-assets/checkpoints/pi05_libero"

    # Seed all random number generators for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Start evaluation
    task_episodes, task_successes = 0, 0
    print(f"\nStarting evaluation: {num_trials} trials (seed={seed})...")


    # Set the JAX compilation cache directory to avoid recompilation and speed up repeated runs.
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))



    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    if task_id >= num_tasks_in_suite:
        raise ValueError(f"Task ID {task_id} is out of range. Task suite has {num_tasks_in_suite} tasks.")

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
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
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
                    element = {
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
                    result = policy.infer(element)
                    action_chunk = result["actions"]
                    assert len(action_chunk) >= REPLAN_STEPS, \
                        f"Policy only predicts {len(action_chunk)} steps, need {REPLAN_STEPS}"
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
            video_filename = f"rollout_task{task_id}_{task_segment}_ep{episode_idx+1}_{suffix}.mp4"
            imageio.mimwrite(
                pathlib.Path(VIDEO_OUT_PATH) / video_filename,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

        # Log progress
        if (episode_idx + 1) % 1== 0:
            print(f"  Episodes: {task_episodes}, Successes: {task_successes} ({task_successes/task_episodes*100:.1f}%)")

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


def create_policy(
    model: _model.BaseModel,
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
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
        norm_stats = _checkpoints.load_norm_stats(pathlib.Path(checkpoint_dir) / "assets", data_config.asset_id)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=False,
        pytorch_device=None,

    )
