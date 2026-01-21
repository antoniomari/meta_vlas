## Training Function
# Suppress all warnings at the very beginning
import warnings
warnings.filterwarnings("ignore")
# Specifically suppress the JAX shape deprecation warning from Flax
warnings.filterwarnings("ignore", message=".*shape requires ndarray or scalar arguments.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax.core.scope")

import contextlib
import copy
import dataclasses
import functools
from typing import Any, Iterator, SupportsIndex, Tuple, Optional, List
import os
import logging
import etils.epath as epath
import time

# Evaluate pretrained model on first task of libero_90
import collections
import math
import pathlib
import imageio
import sys
import random

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'


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
import openpi.shared.download as download


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


def fetch_samples(dataset, idx) -> Tuple[_model.Observation, _model.Actions]:
    examples = [dataset[int(i)] for i in idx]
    batch = _data_loader._collate_fn(examples)
    # Convert to numpy arrays (JAX expects numpy arrays, not PyTorch tensors)
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return np.asarray(x)
        return np.asarray(x)
    batch = jax.tree.map(to_numpy, batch)
    obs, actions = _model.Observation.from_dict(batch), batch["actions"]
    return obs, actions

def load_pi05_libero_model() -> Tuple[_model.BaseModel, _config.TrainConfig]:
    # Load weights from pi0.5 model already fine-tuned on Libero
    checkpoint_gs_path = "gs://openpi-assets/checkpoints/pi05_libero"
    checkpoint_name = "pi05_libero"
    config = _config.get_config("pi05_libero")

    # Download checkpoint
    checkpoint_dir = download.maybe_download(checkpoint_gs_path)
    # Load model
    model = config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    return model, config

@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )



    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # Tells JAX to "donate" (transfer ownership of) the argument at position 1 (partial_params) so its memory can be reused, reducing copies during JIT execution.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


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
    if config.trainable_filter is not None:
        _trainable_filter_for_init = config.trainable_filter
    elif config.freeze_filter is not None:
        _trainable_filter_for_init = nnx.All(nnx.Param, nnx.Not(config.freeze_filter))
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

        # Following marco.py: wrap train state initialization in a function and JIT it
        def init_train_state(params_to_init: at.Params) -> training_utils.TrainState:
            """Initialize train state from params. This function will be JIT-compiled."""
            # Get graphdef for the model (static structure)
            graphdef = nnx.graphdef(model)

            # Apply freeze filter if provided
            params = params_to_init
            if config.freeze_filter is not None:
                params = nnx_utils.state_map(
                    params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
                )

            # Initialize optimizer state on trainable params
            opt_state = tx.init(params.filter(config.trainable_filter))

            # Create and return TrainState
            # Following marco.py pattern: ema_params references same params initially
            return training_utils.TrainState(
                step=0,
                params=params,
                model_def=graphdef,
                tx=tx,
                opt_state=opt_state,
                ema_decay=config.ema_decay,
                ema_params=None if config.ema_decay is None else params,
            )

        # Get initial params for counting
        initial_params = nnx.state(model)
        trainable_params = initial_params.filter(_trainable_filter_for_init)

        # Count trainable parameters
        def param_count(x):
            if hasattr(x, "value"):
                return int(jax.device_get(jnp.size(x.value)))
            return int(jax.device_get(jnp.size(x)))
        param_counts = jax.tree.map(param_count, trainable_params)
        num_trainable_params = sum(jax.tree_util.tree_leaves(param_counts))
        del param_counts, trainable_params
        print(f"Number of trainable parameters: {num_trainable_params:,}")

        # JIT compile the initialization and call it with params
        # Following marco.py: donate_argnums=(0,) donates the params buffer
        train_state = jax.jit(init_train_state, donate_argnums=(0,))(initial_params)

    # Initialize RNG
    rng = jax.random.key(seed)

    # Define trainable_filter for training step
    _trainable_filter = _trainable_filter_for_init

    # JIT compile the training step
    # For TTT, we disable buffer donation since we extract the model immediately after training
    # This prevents "buffer has been deleted or donated" errors when using the model for inference
    if donate_buffers:
        train_step_jit = jax.jit(
            functools.partial(train_step, config),
            donate_argnums=(1,)
        )
    else:
        train_step_jit = jax.jit(
            functools.partial(train_step, config),
            # No buffer donation - buffers will be preserved for model extraction
        )

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

    # Following marco.py: no separate warmup, first training step will compile naturally
    print("Starting training (first step will compile, may take a few minutes)...")

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

        # Training step - aligned with train.py: pass rng, state, batch
        # TIMING: Measure training step (includes GPU computation)
        t0_train = time.perf_counter()
        train_state, info = train_step_jit(rng, train_state, batch)
        jax.block_until_ready(info["loss"])  # Wait for GPU to finish
        t1_train = time.perf_counter()
        train_step_time = (t1_train - t0_train) * 1000  # Convert to ms

        # Logging - aligned with train.py info structure
        loss_val = float(jax.device_get(info["loss"]))
        grad_norm = float(jax.device_get(info["grad_norm"]))
        losses.append(loss_val)  # Store loss for plotting
        infos.append({"loss": loss_val, "grad_norm": grad_norm})

        # Print timing info for every step using tqdm.write to avoid conflicts with progress bar
        total_time = batch_fetch_time + train_step_time
        fetch_percent = (batch_fetch_time / total_time) * 100
        pbar.write(f"Step {step}: fetch={batch_fetch_time:6.2f}ms ({fetch_percent:4.1f}%), train={train_step_time:6.2f}ms, loss={loss_val:.4f}")
        # Note: Detailed timings are printed directly in train_step() when JIT is disabled

        # Update progress bar after every step
        pbar.set_postfix({"loss": f"{loss_val:.4f}", "grad_norm": f"{grad_norm:.4f}"})

        # Debug: Check if parameters are actually changing (only on first few steps after start)
        if step < start_step + 3:
            # Get a sample parameter to check if it's updating
            sample_param = next(iter(jax.tree.leaves(train_state.params.filter(_trainable_filter))))
            if hasattr(sample_param, 'value'):
                param_val = float(jax.device_get(sample_param.value.flat[0]))
                pbar.write(f"  Debug step {step}: sample param value = {param_val:.6f}, grad_norm = {grad_norm:.6f}")

        if step % log_interval == 0 or step == end_step - 1:
            avg_loss = np.mean([info["loss"] for info in infos[-log_interval:]])
            avg_grad_norm = np.mean([info["grad_norm"] for info in infos[-log_interval:]])
            print(f"Step {step}: loss={avg_loss:.4f}, grad_norm={avg_grad_norm:.4f}")

    print(f"\nTraining completed! Final loss: {losses[-1]:.4f}")

    # Return a copy of the trained model from final train_state
    # Block until all computations are done
    jax.block_until_ready(train_state.params)

    # Always create a completely independent copy of params to avoid any buffer sharing issues
    # Even if buffers weren't donated, creating a fresh copy ensures no stale references
    def copy_param(x):
        if isinstance(x, jax.Array):
            # Block until ready to ensure computation is complete
            jax.block_until_ready(x)
            # Force a copy by converting to numpy and back - this creates a completely new buffer
            # This is more reliable than x + 0 which JAX might optimize away
            return jnp.array(np.asarray(x))
        return x

    # Create a completely independent copy of all parameters
    params_copy = jax.tree.map(copy_param, train_state.params)
    # Materialize the copy to ensure all buffers are ready
    jax.block_until_ready(params_copy)

    # Merge to create a fresh model instance with independent buffers
    trained_model = nnx.merge(train_state.model_def, params_copy)

    # Set model to eval mode (not training mode) for inference
    trained_model.eval()

    # Materialize the model to ensure all buffers are ready
    jax.block_until_ready(trained_model)

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
    NUM_STEPS_WAIT = 10
    VIDEO_OUT_PATH = video_out_path
    CHECKPOINT_CONFIG = "pi05_libero"
    CHECKPOINT_DIR = "gs://openpi-assets/checkpoints/pi05_libero"

    # Seed all random number generators for reproducibility
    # Note: Policy RNG must be set during policy creation via create_policy(rng_seed=seed)
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
    CHECKPOINT_CONFIG = "pi05_libero"
    CHECKPOINT_DIR = "gs://openpi-assets/checkpoints/pi05_libero"

    # Seed all random number generators for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Start evaluation
    task_episodes, task_successes = 0, 0
    print(f"\nStarting TTT evaluation: {num_trials} trials (seed={seed})...")
    print(f"TTT settings: {ttt_num_steps} steps every {ttt_frequency} rollout steps")

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

    # Initialize train state for TTT (we'll update the policy's model during TTT)
    train_state = None
    ttt_count = 0

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
        # Store original model state and graphdef at the start of each episode (will be reset at each TTT interval)
        original_model_state = None
        original_model_graphdef = None

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

                    # Perform TTT if at the right frequency
                    if t > 0 and t % ttt_frequency == 0:
                        ttt_count += 1
                        print(f"\n[TTT {ttt_count}] Performing test-time training at step {t}...")

                        # Reset to original model at the start of each TTT interval
                        if original_model_state is not None:
                            print(f"[TTT {ttt_count}] Resetting to original model state...")
                            # Materialize the stored state to ensure it's independent
                            jax.block_until_ready(original_model_state)
                            original_model = nnx.merge(original_model_graphdef, original_model_state)
                            jax.block_until_ready(original_model)
                            policy._model = original_model
                            # Recompile the JIT function with the new model
                            policy._sample_actions = nnx_utils.module_jit(original_model.sample_actions)
                        else:
                            # Store original model state and graphdef at the first TTT
                            print(f"[TTT {ttt_count}] Storing original model state...")
                            # Materialize before storing to ensure we have a snapshot
                            jax.block_until_ready(policy._model)
                            original_model_graphdef = nnx.graphdef(policy._model)
                            original_model_state = nnx.state(policy._model)
                            # Materialize the state to ensure it's a proper snapshot
                            jax.block_until_ready(original_model_state)

                        # Create a copy of the model for TTT
                        print(f"[TTT {ttt_count}] Creating model copy for TTT...")
                        # Materialize current model before copying
                        jax.block_until_ready(policy._model)
                        model_graphdef = nnx.graphdef(policy._model)
                        model_state = nnx.state(policy._model)
                        jax.block_until_ready(model_state)
                        model_copy = nnx.merge(model_graphdef, model_state)
                        jax.block_until_ready(model_copy)

                        # Fetch nearest neighbor indices based on current observation
                        print(f"[TTT {ttt_count}] Fetching {ttt_k} nearest neighbors...")
                        distances, indices, metadata = nn_fetcher.fetch_neighbors(
                            observation=element,
                            use_modalities=ttt_use_modalities,
                            k=ttt_k,
                        )
                        print(f"[TTT {ttt_count}] Retrieved neighbors with similarities: {distances[:min(5, len(distances))]}")
                        obs, actions = fetch_samples(dataset, indices)

                        # Create a simple dataloader that returns the same batch every time
                        class SingleBatchDataLoader:
                            """A simple dataloader that returns the same batch (obs, actions) every time."""
                            def __init__(self, obs: _model.Observation, actions: _model.Actions, data_config: _config.DataConfig):
                                self.obs = obs
                                self.actions = actions
                                self._data_config = data_config

                            def data_config(self) -> _config.DataConfig:
                                return self._data_config

                            def __iter__(self):
                                """Yield the same batch repeatedly."""
                                while True:
                                    yield (self.obs, self.actions)

                        # Get data config from train_config
                        ttt_data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
                        ttt_data_loader = SingleBatchDataLoader(obs, actions, ttt_data_config)

                        # Perform fine-tuning on the copy
                        print(f"[TTT {ttt_count}] Fine-tuning for {ttt_num_steps} steps...")
                        # Disable buffer donation for TTT since we extract the model immediately after
                        trained_model, losses, train_state = train_model_on_fly(
                            model=model_copy,
                            training_data_loader=ttt_data_loader,
                            config=train_config,
                            learning_rate=learning_rate,
                            num_steps=ttt_num_steps,
                            warmup_steps=0,
                            weight_decay=0.0,
                            log_interval=max(1, ttt_num_steps // 2),
                            seed=seed + ttt_count,  # Different seed for each TTT
                            resume_train_state=train_state,
                            resume_losses=None,  # Don't carry over losses
                            donate_buffers=False,  # CRITICAL: Disable buffer donation for TTT
                        )
                        # Print losses
                        print(f"[TTT {ttt_count}] Losses: {losses}")

                        # Update policy with fine-tuned model copy
                        # The trained_model already has independent buffers and is in eval mode
                        # Materialize everything to ensure buffers are ready
                        jax.block_until_ready(trained_model)

                        # Ensure model is in eval mode (should already be set, but double-check)
                        trained_model.eval()

                        # CRITICAL: Before creating the JIT function, we need to ensure the model state
                        # is completely materialized and independent. The key is to force all buffers to CPU
                        # and back to ensure they're completely independent from any training buffers.
                        print(f"[TTT {ttt_count}] Materializing model state for JIT compilation...")

                        # First, ensure trained_model is completely idle
                        jax.block_until_ready(trained_model)

                        # Extract graphdef and state
                        model_graphdef = nnx.graphdef(trained_model)
                        model_state = nnx.state(trained_model)

                        # Materialize all state values by moving to CPU and back - this ensures complete independence
                        def materialize_state(x):
                            if isinstance(x, jax.Array):
                                # Block until ready
                                jax.block_until_ready(x)
                                # Move to CPU (numpy) and back to ensure complete independence
                                # This breaks any potential buffer sharing
                                cpu_value = jax.device_get(x)
                                # Create a completely new JAX array from the CPU value
                                new_array = jnp.array(cpu_value)
                                # Block until the new array is ready
                                jax.block_until_ready(new_array)
                                return new_array
                            return x

                        materialized_state = jax.tree.map(materialize_state, model_state)
                        # Block until all materialization is complete
                        jax.block_until_ready(materialized_state)

                        # Create a fresh model from the materialized state
                        fresh_model = nnx.merge(model_graphdef, materialized_state)
                        fresh_model.eval()
                        # Block until the model is ready
                        jax.block_until_ready(fresh_model)

                        # CRITICAL: Update the policy model and create a fresh JIT function
                        # The key is that fresh_model has completely independent buffers
                        print(f"[TTT {ttt_count}] Updating policy with trained model...")

                        # Update the policy's model
                        policy._model = fresh_model
                        jax.block_until_ready(policy._model)

                        # CRITICAL: Create a completely fresh JIT function with the fresh model
                        # Clear caches first to ensure no stale references
                        jax.clear_caches()

                        # Create the base JIT function - module_jit will capture the fresh model's state
                        base_jit_fn = nnx_utils.module_jit(fresh_model.sample_actions)

                        # Wrap it to ensure outputs are safe to index and convert to numpy
                        # The issue is that outputs might reference buffers that get invalidated
                        # By materializing outputs through numpy, we ensure they're safe
                        def safe_sample_actions(rng, obs, **kwargs):
                            outputs = base_jit_fn(rng, obs, **kwargs)
                            # Block until outputs are ready
                            jax.block_until_ready(outputs)
                            # Materialize outputs by converting to numpy and back to JAX
                            # This creates completely new arrays that don't reference the JIT function's internal buffers
                            materialized_outputs = jax.tree.map(
                                lambda x: jnp.array(np.asarray(x)) if isinstance(x, jax.Array) else x,
                                outputs
                            )
                            jax.block_until_ready(materialized_outputs)
                            return materialized_outputs

                        policy._sample_actions = safe_sample_actions

                        # Block until everything is ready
                        jax.block_until_ready(policy._model)

                        print(f"[TTT {ttt_count}] Fine-tuning complete. Final loss: {losses[-1]:.4f}\n")

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
                import traceback
                traceback.print_exc()
                break

        task_episodes += 1

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
            print(f"  Episodes: {task_episodes}, Successes: {task_successes} ({task_successes/task_episodes*100:.1f}%)")

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
        norm_stats = _checkpoints.load_norm_stats(pathlib.Path(checkpoint_dir) / "assets", data_config.asset_id)

    # Create RNG key if seed provided for reproducible sampling
    rng_key = jax.random.PRNGKey(rng_seed) if rng_seed is not None else None

    return _policy.Policy(
        model,
        # rng=rng_key, NOTE: disabled, this is bugged in OpenPI
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
