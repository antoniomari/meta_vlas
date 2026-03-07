## Training Function
# Suppress all warnings at the very beginning
import warnings

warnings.filterwarnings("ignore")
# Specifically suppress the JAX shape deprecation warning from Flax
warnings.filterwarnings(
    "ignore", message=".*shape requires ndarray or scalar arguments.*"
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax.core.scope")

import dataclasses
import functools
from typing import Any, Iterator, SupportsIndex, Tuple, Optional, List, Callable
from collections import defaultdict

from meta_libero.src.nn_fetcher import NearestNeighborFetcher
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
from pathlib import Path
import imageio
from PIL import Image, ImageDraw, ImageFont
import sys
import random
import h5py

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops"

from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
import openpi.models.pi0_config as pi0_config
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "third_party" / "libero"))
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi.policies import policy_config as _policy_config
from openpi.policies import policy as _policy
from openpi.policies import libero_policy as _openpi_libero_policy
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
import torch

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

from meta_libero.src.rendering import (
    plot_observation_with_decoded_prompt,
    decode_tokenized_prompt,
    make_observation_from_simulator,
    _draw_step_on_frame,
    render_image,
)
from meta_libero.src.dataset import FilteredDataset
from meta_libero.src.libero_inputs_override import (
    BatchableLiberoInputs,
    BatchableLiberoOutputs,
)
from meta_libero.src.utils import (
    get_gpu_memory_usage,
    print_memory_checkpoint,
    _quat2axisangle,
    _get_libero_env,
    load_pi05_libero_model,
    fetch_samples,
    compute_alignment_ratio,
    _compute_aux_denoising_losses,
    nn_lookup,
    create_policy,
    new_policy_like,
)
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

# Global flag to disable JIT (set to True to disable JIT compilation)
DISABLE_JIT = False
PRINT_TIME_EXECUTION = False



def _legacy_fetch_samples(dataset, idx, repeat=1) -> Tuple[_model.Observation, _model.Actions]:
    def _find_filtered_dataset(ds):
        cur = ds
        while True:
            if isinstance(cur, FilteredDataset):
                return cur
            if hasattr(cur, "_dataset"):
                cur = cur._dataset
                continue
            return None

    filtered_ds = _find_filtered_dataset(dataset)
    idx_int = [int(i) for i in idx]

    if filtered_ds is not None:
        print(f"Filtered dataset found: {filtered_ds}")
        # NN indices are physical/global; map them to logical indices within FilteredDataset.
        physical_to_logical = {int(physical_idx): logical_idx for logical_idx, physical_idx in enumerate(filtered_ds.indices)}
        mapped = [physical_to_logical[i] for i in idx_int]
        # Corrected mapping: for each physical NN index i, get logical index from physical_to_logical, then fetch from dataset
        examples = [dataset[mapped_i] for mapped_i in mapped]
    else:
        examples = [dataset[i] for i in idx_int]

    examples = [example for _ in range(repeat) for example in examples]
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


def _legacy_load_pi05_libero_model(
    use_base_model: bool = False,
    use_lora: bool = False,
    action_expert_only: bool = False
) -> Tuple[_model.BaseModel, _config.TrainConfig]:
    # Load weights from pi0.5 model already fine-tuned on Libero

    if use_base_model:
        checkpoint_gs_path = "gs://openpi-assets/checkpoints/pi05_base"
    else:
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
        # Note: I set train=False to avoid preprocessing observation
        # TODO: consider what is the best thing to do here
        chunked_loss = model.compute_loss(rng, observation, actions, train=False)
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
    # Optional auxiliary denoising-loss evaluation on each optimization step.
    aux_observation: _model.Observation | None = None,
    aux_actions: _model.Actions | None = None,
    aux_num_samples: int = 10,
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
    train_step_jit = functools.partial(train_step, config)

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
    aux_losses_by_step: list[list[float]] = []
    aux_enabled = aux_observation is not None and aux_actions is not None
    if aux_enabled and aux_num_samples < 1:
        raise ValueError(f"aux_num_samples must be >= 1, got {aux_num_samples}")

    # Disable progress bar for train_model_on_fly (called during TTT, progress shown in main evaluation loop)
    # Train from start_step to end_step (num_steps additional steps)
    pbar = tqdm(
        range(start_step, end_step),
        desc="Training",
        total=num_steps,  # Show progress as num_steps (the additional steps)
        dynamic_ncols=True,
        mininterval=0.5,
        maxinterval=2.0,
        # disable=True,  # Disable progress bar for TTT training
    )

    # Initialize iterator
    data_iter = iter(training_data_loader)

    # Following marco.py: no separate warmup, first training step will compile naturally
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

        # Optional: compute auxiliary denoising loss after each optimization step.
        if aux_enabled:
            step_model = nnx.merge(train_state.model_def, train_state.params)
            step_model.eval()
            step_aux_losses = _compute_aux_denoising_losses(
                model=step_model,
                observation=aux_observation,
                actions=aux_actions,
                seed=seed,
                num_samples=aux_num_samples,
            )
            aux_losses_by_step.append(step_aux_losses)
            pbar.write(
                f"Step {step}: aux denoising losses={step_aux_losses} "
                f"(mean={float(np.mean(step_aux_losses)):.4f})"
            )
            del step_model

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


        if step % log_interval == 0 or step == end_step - 1:
            avg_loss = np.mean([info["loss"] for info in infos[-log_interval:]])
            avg_grad_norm = np.mean(
                [info["grad_norm"] for info in infos[-log_interval:]]
            )
            pbar.write(f"Step {step}: loss={avg_loss:.4f}, grad_norm={avg_grad_norm:.4f}")

    # Merge to create a fresh model instance with independent buffers
    trained_model = nnx.merge(train_state.model_def, train_state.params)
    trained_model.eval()

    # Expose per-step auxiliary losses for callers without changing return signature.
    train_model_on_fly.last_aux_losses_by_step = aux_losses_by_step  # type: ignore[attr-defined]

    return trained_model, losses, train_state


def _legacy_quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _legacy_get_libero_env(task, resolution, seed):
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
    train_config: _config.TrainConfig,
    num_trials: int = 10,
    task_suite_name: str = "libero_90",
    task_id: int = 0,
    num_steps_wait: int = 10,
    save_video: bool = True,
    video_out_path: str = "data/libero/videos",
    seed: int = 0,
    plot_observations: bool = False,
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
    return run_evaluation_ttt(
        policy=policy,
        nn_fetcher=None,
        train_config=train_config,
        dataset=None,
        num_trials=num_trials,
        task_suite_name=task_suite_name,
        task_id=task_id,
        num_steps_wait=num_steps_wait,
        save_video=save_video,
        video_out_path=video_out_path,
        seed=seed,
        ttt_num_steps=0,
        ttt_frequency=1000,
        ttt_k=1,
        plot_observations=plot_observations,
        disable_adaptation=True,
        )


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
    max_ttt_step: int = 1000, # No further TTT after this step
    ttt_frequency: int = 50,
    learning_rate: float = 2.5e-5,
    ttt_k: int = 50,
    ttt_use_modalities: Optional[List[str]] = None,
    plot_observations: bool = False,
    disable_adaptation: bool = False,
    repeat_batch: int = 1,
    reset_policy: bool = True,
    use_test_task: bool = False,
    random_neighbors: bool = False,
    cfg_weight: float = 1.0,
    num_samples: int = 10,
    debug_metrics: bool = False,
):


    return run_evaluation_with_adaptation(
        adaptation_fn=adapt_fn_ttt,
        policy=policy,
        nn_fetcher=nn_fetcher,
        train_config=train_config,
        dataset=dataset,
        num_trials=num_trials,
        task_suite_name=task_suite_name,
        task_id=task_id,
        num_steps_wait=num_steps_wait,
        save_video=save_video,
        video_out_path=video_out_path,
        seed=seed,
        ttt_num_steps=ttt_num_steps,
        max_ttt_step=max_ttt_step,
        ttt_frequency=ttt_frequency,
        learning_rate=learning_rate,
        ttt_k=ttt_k,
        ttt_use_modalities=ttt_use_modalities,
        plot_observations=plot_observations,
        repeat_batch=repeat_batch,
        reset_policy=reset_policy,
        use_test_task=use_test_task,
        random_neighbors=random_neighbors,
        cfg_weight=cfg_weight,
        num_samples=num_samples,
        debug_metrics=debug_metrics,
        adapt_kwargs={},
        disable_adaptation=disable_adaptation,
    )


def run_evaluation_noise(
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
    max_noise_step: int = 150, # No further noise after this step
    noise_frequency: int = 50,
    noise_sigma: float = 0.01,
    plot_observations: bool = False,
    reset_policy: bool = True,
):

    return run_evaluation_with_adaptation(
        adaptation_fn=adapt_fn_gaussian_perturbation,
        policy=policy,
        nn_fetcher=nn_fetcher,
        train_config=train_config,
        dataset=dataset,
        num_trials=num_trials,
        task_suite_name=task_suite_name,
        task_id=task_id,
        num_steps_wait=num_steps_wait,
        save_video=save_video,
        video_out_path=video_out_path,
        seed=seed,
        ttt_num_steps=0,
        max_ttt_step=max_noise_step,
        ttt_frequency=noise_frequency,
        ttt_k=1,
        plot_observations=plot_observations,
        repeat_batch=1,
        reset_policy=reset_policy,
        adapt_kwargs=dict(noise_std=noise_sigma),
    )


def _infer_action_chunk_samples(
    policy_obj: _policy.Policy,
    obs_dict: dict[str, Any],
    noise_arr: np.ndarray | jnp.ndarray | None,
) -> _model.Actions:
    """Stable per-sample inference helper used by rollout and TTT paths."""
    if noise_arr is None or getattr(noise_arr, "ndim", 0) <= 2:
        action_chunk_single = policy_obj.infer(obs_dict, noise=noise_arr)["actions"]
        if getattr(action_chunk_single, "ndim", 0) == 2:
            action_chunk_single = action_chunk_single[None, ...]
        return action_chunk_single

    sample_count = int(noise_arr.shape[0])
    action_chunks = []
    for sample_idx in range(sample_count):
        action_i = policy_obj.infer(obs_dict, noise=noise_arr[sample_idx])["actions"]
        action_i = action_i if isinstance(action_i, jax.Array) else jnp.asarray(action_i)
        action_chunks.append(action_i)
    return jnp.stack(action_chunks, axis=0)


def _max_steps_for_task_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220  # longest training demo has 193 steps
    if task_suite_name == "libero_object":
        return 280  # longest training demo has 254 steps
    if task_suite_name == "libero_goal":
        return 300  # longest training demo has 270 steps
    if task_suite_name == "libero_10":
        return 520  # longest training demo has 505 steps
    if task_suite_name == "libero_90":
        return 400  # longest training demo has 373 steps
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _build_curr_obs_dict(
    obs: dict[str, Any],
    img: np.ndarray,
    wrist_img: np.ndarray,
    task_description: str,
) -> dict[str, Any]:
    return {
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


def _preprocess_images(
    obs: dict[str, Any],
    *,
    resize_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img_resized = image_tools.resize_with_pad(img, resize_size, resize_size)
    wrist_img_resized = image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    return image_tools.convert_to_uint8(img_resized), image_tools.convert_to_uint8(wrist_img_resized)


def _plot_ttt_debug_metrics(
    *,
    losses: list[float],
    distances_actions: list[tuple[int, float]],
    ttt_count: int,
) -> None:
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


def _plot_alignment_ratio_by_step(alignment_ratio_by_step: list[tuple[int, float]]) -> None:
    if not alignment_ratio_by_step:
        return
    steps, ratios = zip(*alignment_ratio_by_step)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, ratios, "b-o", markersize=3)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.5, label="0.05")
    ax.axhline(0.2, color="gold", linestyle="--", linewidth=1.5, label="0.2")
    ax.set_title("Alignment ratio")
    ax.set_xlabel("Step")
    ax.set_ylabel("Alignment ratio")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def _save_episode_losses_plot(
    *,
    episode_losses: list[list[float]],
    ttt_num_steps: int,
    video_out_path: str,
    episode_idx: int,
    done: bool,
) -> None:
    if not episode_losses or ttt_num_steps <= 1:
        return
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
    os.makedirs(pathlib.Path(video_out_path).parent / "plots_losses", exist_ok=True)
    losses_plot_path = pathlib.Path(video_out_path).parent / "plots_losses" / f"losses_ep_{episode_idx+1}.pdf"
    plt.savefig(losses_plot_path, bbox_inches="tight")
    plt.close(fig)


def _save_rollout_video(
    *,
    save_video: bool,
    video_out_path: str,
    task_id: int,
    task_description: str,
    episode_idx: int,
    done: bool,
    replay_images: list[np.ndarray],
) -> None:
    if not save_video:
        return
    suffix = "success" if done else "failure"
    task_segment = task_description.replace(" ", "_")
    video_filename = f"rollout_ttt_task{task_id}_{task_segment}_ep{episode_idx+1}_{suffix}.mp4"
    imageio.mimwrite(
        pathlib.Path(video_out_path) / video_filename,
        [np.asarray(x) for x in replay_images],
        fps=10,
    )


def _run_single_episode_with_adaptation(
    *,
    env: Any,
    policy: _policy.Policy,
    original_model: _model.BaseModel,
    initial_state: Any,
    episode_seed: int,
    task_description: str,
    max_steps: int,
    num_steps_wait: int,
    libero_dummy_action: list[float],
    resize_size: int,
    replan_steps: int,
    adaptation_fn: Callable,
    train_config: _config.TrainConfig,
    ttt_data_config: _config.DataConfig,
    nn_fetcher: Any,
    dataset: Any,
    ttt_use_modalities: Optional[List[str]],
    ttt_k: int,
    repeat_batch: int,
    plot_observations: bool,
    use_test_task: bool,
    random_neighbors: bool,
    disable_adaptation: bool,
    ttt_frequency: int,
    max_ttt_step: int,
    ttt_num_steps: int,
    learning_rate: float,
    num_samples: int,
    debug_metrics: bool,
    adapt_kwargs: dict[str, Any],
    reset_policy: bool,
    cfg_weight: float,
    ttt_count: int,
    jax_key: jax.Array,
    pbar: Any,
) -> tuple[bool, int, list[np.ndarray], list[tuple[int, float]], list[tuple[int, float]], list[list[float]], list[tuple[int, float]], int, _policy.Policy, jax.Array]:
    env.seed(episode_seed)
    env.reset()

    action_plan = collections.deque()
    obs = env.set_init_state(initial_state)
    done = False
    t = 0

    raw_img_sum = float(np.sum(obs["agentview_image"]))
    raw_wrist_img_sum = float(np.sum(obs["robot0_eye_in_hand_image"]))
    raw_img_mean = float(np.mean(obs["agentview_image"]))
    raw_wrist_img_mean = float(np.mean(obs["robot0_eye_in_hand_image"]))

    check_determinism = False
    if check_determinism:
        pbar.write(
            f"[ENV CHECK] Raw env images START - base: sum={raw_img_sum:.10f}, mean={raw_img_mean:.10f}, "
            f"wrist: sum={raw_wrist_img_sum:.10f}, mean={raw_wrist_img_mean:.10f}"
        )

    replay_images: list[np.ndarray] = []
    distances_actions: list[tuple[int, float]] = []
    similarities: list[tuple[int, float]] = []
    alignment_ratio_by_step: list[tuple[int, float]] = []
    episode_losses: list[list[float]] = []

    policy = new_policy_like(policy, original_model)

    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, reward, done, info = env.step(libero_dummy_action)
            t += 1
            continue

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

        img, wrist_img = _preprocess_images(obs, resize_size=resize_size)

        if should_check_images:
            uint8_img_sum = float(np.sum(img))
            uint8_wrist_img_sum = float(np.sum(wrist_img))
            uint8_img_mean = float(np.mean(img))
            uint8_wrist_img_mean = float(np.mean(wrist_img))
            pbar.write(
                f"[ENV CHECK] After uint8 (t={t}) - base: sum={uint8_img_sum:.10f}, mean={uint8_img_mean:.10f}, "
                f"wrist: sum={uint8_wrist_img_sum:.10f}, mean={uint8_wrist_img_mean:.10f}"
            )

        replay_images.append(_draw_step_on_frame(img, t))

        if not action_plan:
            curr_obs_dict = _build_curr_obs_dict(obs, img, wrist_img, task_description)
            policy._rng, rng = jax.random.split(policy._rng)
            if num_samples > 1:
                noise = jax.random.normal(
                    rng, (num_samples, original_model.action_horizon, original_model.action_dim)
                )
            else:
                noise = jax.random.normal(
                    rng, (original_model.action_horizon, original_model.action_dim)
                )

            # Sample action chunk from policy
            action_chunk_samples = _infer_action_chunk_samples(policy, curr_obs_dict, noise)
            action_chunk = action_chunk_samples[0]

            # Compute alignment ratio between action chunk and the action the model would have sampled with empty prompt
            # action_chunk_cfg is the action taken with classifier free guidance ()
            alignment_ratio = compute_alignment_ratio(
                policy, action_chunk_samples, curr_obs_dict, noise=noise, cfg_weight=cfg_weight
            )
            pbar.write(f"[TTT] Alignment ratio: {alignment_ratio:.4f}")
            alignment_ratio_by_step.append((t, alignment_ratio))


            assert len(action_chunk) >= replan_steps, (
                f"Policy only predicts {len(action_chunk)} steps, need {replan_steps}"
            )

            if check_determinism:
                    action_chunk_jax_check = (
                        action_chunk if isinstance(action_chunk, jax.Array) else jnp.array(action_chunk)
                    )
                    action_chunk_before_ttt_sum = float(jax.device_get(jnp.sum(action_chunk_jax_check)))
                    action_chunk_before_ttt_mean = float(jax.device_get(jnp.mean(action_chunk_jax_check)))
                    action_chunk_before_ttt_norm = float(jax.device_get(jnp.linalg.norm(action_chunk_jax_check)))
                    pbar.write(
                        f"[TTT] Action chunk BEFORE TTT (t={t}) - sum: {action_chunk_before_ttt_sum:.10f}, "
                        f"mean: {action_chunk_before_ttt_mean:.10f}, norm: {action_chunk_before_ttt_norm:.10f}"
                    )

            if not disable_adaptation and (t - num_steps_wait) % ttt_frequency == 0 and t > max_ttt_step:
                observation = make_observation_from_simulator(policy, curr_obs_dict)
                obs_ttt, actions_fetched, distances = nn_lookup(
                    observation=observation,
                    nn_fetcher=nn_fetcher,
                    dataset=dataset,
                    use_modalities=ttt_use_modalities,
                    k=ttt_k,
                    repeat_batch=repeat_batch,
                    plot_observations=plot_observations,
                    use_test_task=use_test_task,
                    pbar=pbar,
                )
                del obs_ttt, actions_fetched, distances

            if not disable_adaptation and (t - num_steps_wait) % ttt_frequency == 0 and t <= max_ttt_step:
                ttt_count += 1
                pbar.write(f"[TTT {ttt_count}] Starting TTT update")

                if check_determinism:
                    jax_key, jax_subkey = jax.random.split(jax_key)
                    jax_random_val = float(jax.device_get(jax.random.uniform(jax_subkey)))
                    python_random_val = random.random()
                    numpy_random_val = np.random.rand()
                    pbar.write(
                        f"[TTT {ttt_count}] Determinism check - JAX: {jax_random_val:.10f}, "
                        f"Python: {python_random_val:.10f}, NumPy: {numpy_random_val:.10f}, "
                        f"t={t}"
                    )

                observation = make_observation_from_simulator(policy, curr_obs_dict)
                obs_ttt, actions_fetched, distances = nn_lookup(
                    observation=observation,
                    nn_fetcher=nn_fetcher,
                    dataset=dataset,
                    use_modalities=ttt_use_modalities,
                    k=ttt_k,
                    repeat_batch=repeat_batch,
                    plot_observations=plot_observations,
                    use_test_task=use_test_task,
                    pbar=pbar,
                    random_neighbors=random_neighbors,
                )

                trained_model, losses = adaptation_fn(
                    policy=policy,
                    train_config=train_config,
                    ttt_data_config=ttt_data_config,
                    train_obs=obs_ttt,
                    train_actions=actions_fetched,
                    learning_rate=learning_rate,
                    num_steps=ttt_num_steps,
                    warmup_steps=0,
                    weight_decay=0.0,
                    log_interval=max(1, ttt_num_steps // 2),
                    seed=episode_seed + ttt_count,
                    pbar=pbar,
                    num_samples=num_samples,
                    debug_metrics=debug_metrics,
                    **adapt_kwargs,
                )

                ttt_policy = new_policy_like(policy, trained_model)
                ttt_action_chunk_samples = _infer_action_chunk_samples(ttt_policy, curr_obs_dict, noise)


                # Compute distance between action chunk and ttt action chunk

                action_chunk_jax = (
                    action_chunk_samples if isinstance(action_chunk_samples, jax.Array) else jnp.array(action_chunk_samples)
                )
                ttt_action_chunk_jax = (
                    ttt_action_chunk_samples if isinstance(ttt_action_chunk_samples, jax.Array) else jnp.array(ttt_action_chunk_samples)
                )
                action_distance_per_sample = jnp.linalg.norm(action_chunk_jax - ttt_action_chunk_jax, axis=(1, 2))
                action_distance = float(jax.device_get(jnp.mean(action_distance_per_sample)))

                distances_actions.append((t, action_distance))
                similarities.append((t, distances[0]))
                episode_losses.append(list(losses))
                pbar.write(
                    f"[TTT {ttt_count}] Distance between action_chunk and actions_fetched: {action_distance:.4f}"
                )

                if plot_observations:
                    _plot_ttt_debug_metrics(
                        losses=list(losses),
                        distances_actions=distances_actions,
                        ttt_count=ttt_count,
                    )

                # Use the ttt action chunk as the action chunk for the next step
                action_chunk = ttt_action_chunk_samples[0]

                if reset_policy:
                    del trained_model
                    print_memory_checkpoint(f"[TTT {ttt_count}] After deleting trained model", 1299)
                else:
                    policy = new_policy_like(policy, trained_model)

                del losses, obs_ttt, actions_fetched, ttt_policy
                print_memory_checkpoint(
                    f"[TTT {ttt_count}] At the end of the TTT update deleting objects", 1300
                )

            action_plan.extend(action_chunk[:replan_steps])

        action = action_plan.popleft()
        obs, reward, done, info = env.step(action.tolist())
        if done:
            break
        t += 1

    return (
        bool(done),
        t,
        replay_images,
        distances_actions,
        similarities,
        episode_losses,
        alignment_ratio_by_step,
        ttt_count,
        policy,
        jax_key,
    )


def run_evaluation_with_adaptation(
    policy: _policy.Policy,
    nn_fetcher: Any,
    train_config: _config.TrainConfig,
    dataset: Any,
    adaptation_fn: Callable,
    num_trials: int = 10,
    task_suite_name: str = "libero_90",
    task_id: int = 0,
    num_steps_wait: int = 10,
    save_video: bool = True,
    video_out_path: str = "data/libero/videos",
    seed: int = 0,
    ttt_num_steps: int = 10,
    max_ttt_step: int = 1000, # No further TTT after this step
    ttt_frequency: int = 50,
    learning_rate: float = 2.5e-5,
    ttt_k: int = 50,
    ttt_use_modalities: Optional[List[str]] = None,
    plot_observations: bool = False,
    repeat_batch: int = 1,
    reset_policy: bool = True,
    use_test_task: bool = False,
    random_neighbors: bool = False,
    cfg_weight: float = 1.0,
    num_samples: int = 10,
    debug_metrics: bool = False,
    adapt_kwargs: dict[str, Any] = {},
    disable_adaptation: bool = False,
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

    #########################################################
    # Part 1: Initialize the environment and the policy
    #########################################################
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    LIBERO_ENV_RESOLUTION = 224 if task_suite_name == "libero_90" else 256
    RESIZE_SIZE = 224
    REPLAN_STEPS = 5
    VIDEO_OUT_PATH = video_out_path

    # Seed all random number generators for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    jax_key = jax.random.PRNGKey(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

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

    max_steps = _max_steps_for_task_suite(task_suite_name)

    if save_video:
        pathlib.Path(VIDEO_OUT_PATH).mkdir(parents=True, exist_ok=True)

    # Get task
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    # Initialize environment
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
    print(f"Task: {task_description}")

    ttt_count = 0
    original_model = policy._model
    ttt_data_config = train_config.data.create(
        train_config.assets_dirs, train_config.model
    )

    print_trainable_parameters(nnx.state(original_model), train_config.trainable_filter)


    #########################################################
    # Part 2: Run evaluation episodes
    #########################################################

    # Run evaluation episodes
    pbar = tqdm(range(num_trials), desc=f"Task {task_id} | Success: 0/0 (0.0%)")
    for episode_idx in pbar:
        pbar.write(f"Episode {episode_idx+1} of {num_trials}")
        (
            done,
            t,
            replay_images,
            distances_actions,
            similarities,
            episode_losses,
            alignment_ratio_by_step,
            ttt_count,
            policy,
            jax_key,
        ) = _run_single_episode_with_adaptation(
            env=env,
            policy=policy,
            original_model=original_model,
            initial_state=initial_states[episode_idx],
            episode_seed=seed + episode_idx,
            task_description=task_description,
            max_steps=max_steps,
            num_steps_wait=num_steps_wait,
            libero_dummy_action=LIBERO_DUMMY_ACTION,
            resize_size=RESIZE_SIZE,
            replan_steps=REPLAN_STEPS,
            adaptation_fn=adaptation_fn,
            train_config=train_config,
            ttt_data_config=ttt_data_config,
            nn_fetcher=nn_fetcher,
            dataset=dataset,
            ttt_use_modalities=ttt_use_modalities,
            ttt_k=ttt_k,
            repeat_batch=repeat_batch,
            plot_observations=plot_observations,
            use_test_task=use_test_task,
            random_neighbors=random_neighbors,
            disable_adaptation=disable_adaptation,
            ttt_frequency=ttt_frequency,
            max_ttt_step=max_ttt_step,
            ttt_num_steps=ttt_num_steps,
            learning_rate=learning_rate,
            num_samples=num_samples,
            debug_metrics=debug_metrics,
            adapt_kwargs=adapt_kwargs,
            reset_policy=reset_policy,
            cfg_weight=cfg_weight,
            ttt_count=ttt_count,
            jax_key=jax_key,
            pbar=pbar,
        )

        task_episodes += 1
        if done:
            task_successes += 1

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
                "num_steps": t,
                "alignment_ratio_by_step": list(alignment_ratio_by_step),
            }
        )

        _plot_alignment_ratio_by_step(list(alignment_ratio_by_step))
        _save_episode_losses_plot(
            episode_losses=episode_losses,
            ttt_num_steps=ttt_num_steps,
            video_out_path=VIDEO_OUT_PATH,
            episode_idx=episode_idx,
            done=done,
        )
        _save_rollout_video(
            save_video=save_video,
            video_out_path=VIDEO_OUT_PATH,
            task_id=task_id,
            task_description=task_description,
            episode_idx=episode_idx,
            done=done,
            replay_images=replay_images,
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

    return success_rate, all_episode_metrics


def _legacy_new_policy_like(policy: _policy.Policy, model: _model.BaseModel):

    return _policy.Policy(
        model,
        # rng=rng_key, NOTE: disabled, this is bugged in OpenPI
        transforms=policy._input_transform.transforms,
        output_transforms=policy._output_transform.transforms,
        sample_kwargs=policy._sample_kwargs,
        metadata=policy._metadata,
        is_pytorch=policy._is_pytorch_model,
        pytorch_device=policy._pytorch_device,
    )


def _legacy_create_policy(
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
    libero_inputs_cls = _openpi_libero_policy.LiberoInputs
    libero_outputs_cls = _openpi_libero_policy.LiberoOutputs
    if _openpi_libero_policy.LiberoInputs is not BatchableLiberoInputs:
        _openpi_libero_policy.LiberoInputs = BatchableLiberoInputs
    if _openpi_libero_policy.LiberoOutputs is not BatchableLiberoOutputs:
        _openpi_libero_policy.LiberoOutputs = BatchableLiberoOutputs

    # TODO: check how to provide data_config here
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    replaced_inputs = []
    for tfm in data_config.data_transforms.inputs:
        if isinstance(tfm, libero_inputs_cls):
            replaced_inputs.append(BatchableLiberoInputs(model_type=tfm.model_type))
        else:
            replaced_inputs.append(tfm)
    replaced_outputs = []
    for tfm in data_config.data_transforms.outputs:
        if isinstance(tfm, libero_outputs_cls):
            replaced_outputs.append(BatchableLiberoOutputs())
        else:
            replaced_outputs.append(tfm)
    data_config = dataclasses.replace(
        data_config,
        data_transforms=transforms.Group(
            inputs=tuple(replaced_inputs),
            outputs=tuple(replaced_outputs),
        ),
    )

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



def _legacy_nn_lookup(
    observation: _model.Observation,
    nn_fetcher: NearestNeighborFetcher,
    dataset: Any,
    pbar: tqdm,
    use_modalities: Optional[List[str]],
    k: int,
    repeat_batch: int,
    plot_observations: bool,
    use_test_task: bool = False,
    random_neighbors: bool = False,
):
    start_time = time.time()

    # For libero90 nn_fetcher, we need to mirror horizontally the images
    if nn_fetcher.normalize_per_modality: # True for libero90, False for libero10
        observation.images["base_0_rgb"] = observation.images["base_0_rgb"][:, :, ::-1, :]
        observation.images["left_wrist_0_rgb"] = observation.images["left_wrist_0_rgb"][:, :, ::-1, :]

    print(f"Using modalities: {use_modalities}")

    distances, indices, metadata = nn_fetcher.fetch_neighbors(
        observation=observation,
        use_modalities=use_modalities,
        filter_text_first=True,
        k=k,
    )
    end_time = time.time()
    pbar.write(
        f"\t({end_time - start_time:.2f}s) Retrieved neighbors with similarities: {distances[:min(5, len(distances))]}"
    )

    if random_neighbors:
        indices = np.random.choice(len(dataset), size=k, replace=False)

    nn_observations, nn_actions = fetch_samples(dataset, indices, repeat=repeat_batch)

    # Copy and repeat the tokenized prompt fields from query observation
    if use_test_task:
        nn_observations = dataclasses.replace(
            nn_observations,
            tokenized_prompt=jnp.repeat(observation.tokenized_prompt, k, axis=0),
            tokenized_prompt_mask=jnp.repeat(observation.tokenized_prompt_mask, k, axis=0),
            token_ar_mask=jnp.repeat(observation.token_ar_mask, k, axis=0) if observation.token_ar_mask is not None else None,
            token_loss_mask=jnp.repeat(observation.token_loss_mask, k, axis=0) if observation.token_loss_mask is not None else None,
        )

    # Plot current observation
    if plot_observations:
        plot_observation_with_decoded_prompt(
            observation=observation,
            similarity_score=None,
            plot_title_prefix=f"\tCurrent Observation",
        )

        # Plot fetched observation (first/best neighbor)
        # Note: obs might be a batch, but we'll plot the first one which is the best match
        if len(indices) > 0 and len(distances) > 0:
            # Extract first observation from batch if needed
            # If obs is a batch, plot_observation_images should handle extracting the first element
            plot_observation_with_decoded_prompt(
                observation=nn_observations,
                similarity_score=distances,
                plot_title_prefix=f"\tFetched Observation (Best Match)",
            )

    return nn_observations, nn_actions, distances


def _legacy_compute_aux_denoising_losses(
    model: _model.BaseModel,
    observation: _model.Observation,
    actions: _model.Actions,
    *,
    seed: int,
    num_samples: int,
) -> list[float]:
    """
    Compute auxiliary denoising losses for logging only.

    This does NOT perform optimization; it evaluates model.compute_loss multiple
    times with different RNG seeds.
    """
    losses: list[float] = []
    for i in range(num_samples):
        rng = jax.random.PRNGKey(seed + i)
        chunked_loss = model.compute_loss(rng, observation, actions, train=False)
        losses.append(float(jax.device_get(jnp.mean(chunked_loss))))
    return losses


def adapt_fn_ttt(
    policy: _policy.Policy,
    train_config: _config.TrainConfig,
    ttt_data_config: _config.DataConfig,
    train_obs: _model.Observation,
    train_actions: _model.Actions,
    learning_rate: float,
    num_steps: int,
    warmup_steps: int,
    weight_decay: float,
    log_interval: int,
    seed: int,
    pbar: tqdm,
    **kwargs,
):
    # Prepare dataloader for TTT
    start_time = time.time()

    ttt_data_loader = SingleBatchDataLoader(
        train_obs, train_actions, ttt_data_config
    )
    end_time = time.time()
    pbar.write(
        f"\t({end_time - start_time:.2f}s) Prepared dataloader"
    )

    # Debug: Check if train_config object is the same between TTT calls

    # Restore original model before TTT so each TTT starts from the same baseline
    print_memory_checkpoint(
        f"\tBefore copying original model", 1288
    )
    time_start = time.time()
    model_copy = copy_model(policy._model, train_config)
    time_end = time.time()
    pbar.write(
        f"\t({time_end - time_start:.2f}s) Copied original model"
    )
    print_memory_checkpoint(
        f"\tAfter copying original model", 1290
    )
    # NOTE: here we go 6.25 GB -> 12.58 GB as the model is copied

    # Perform fine-tuning on the copy (with donation enabled to save memory)
    # Each TTT should start fresh - don't resume from previous TTT state to avoid memory accumulation
    # Use donation=True to save memory - the copy will be modified, original is preserved
    debug_metrics = bool(kwargs.get("debug_metrics", False))
    num_samples = int(kwargs.get("num_samples", kwargs.get("loss_samples", 10)))
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    trained_model, train_losses, train_state = train_model_on_fly(
        model=model_copy,  # Pass copy, original model is preserved
        training_data_loader=ttt_data_loader,
        config=train_config,
        learning_rate=learning_rate,
        num_steps=num_steps,
        warmup_steps=0,
        weight_decay=0.0,
        log_interval=max(1, num_steps // 2),
        seed=seed,
        resume_train_state=None,  # Each TTT starts fresh - don't accumulate optimizer state
        resume_losses=None,  # Don't carry over losses
        donate_buffers=True,  # Enable donation to save memory (copy is modified, original preserved)
        aux_observation=train_obs if debug_metrics else None,
        aux_actions=train_actions if debug_metrics else None,
        aux_num_samples=num_samples,
    )

    # Use fine-tuned model to generate new action plan
    # NOTE: No need to block - trained_model was already returned from train_model_on_fly
    # which handled all necessary blocking during the copy/merge process
    trained_model.eval()

    # Auxiliary denoising losses from every inner training step (debug only).
    if debug_metrics:
        aux_losses_by_step = getattr(train_model_on_fly, "last_aux_losses_by_step", [])
        aux_losses = [float(np.mean(step_losses)) for step_losses in aux_losses_by_step]
        if aux_losses_by_step:
            pbar.write(f"\tAux denoising losses per inner step: {aux_losses_by_step}")
            pbar.write(f"\tAux denoising mean per inner step: {aux_losses}")
    else:
        aux_losses = []

    del train_state, ttt_data_loader
    del train_losses

    return trained_model, aux_losses


def adapt_fn_gaussian_perturbation(
    policy: _policy.Policy,
    train_config: _config.TrainConfig,
    pbar: tqdm,
    noise_std: float = 0.01,
    seed: int = 0,
    **kwargs
):
    """
    Adapt the model by adding Gaussian noise to all trainable parameters.

    This is a simple baseline adaptation method that perturbs trainable parameters
    with Gaussian noise instea[float]d of performing gradient-based optimization.

    Args:
        policy: The policy containing the model to adapt
        train_config: Training configuration with trainable_filter
        ttt_data_config: Data configuration (unused, kept for API consistency)
        train_obs: Training observation (unused, kept for API consistency)
        train_actions: Training actions (unused, kept for API consistency)
        fetched_observation: Fetched observation (unused, kept for API consistency)
        actions_fetched: Fetched actions (unused, kept for API consistency)
        learning_rate: Learning rate (unused, kept for API consistency)
        num_steps: Number of steps (unused, kept for API consistency)
        noise_std: Standard deviation of Gaussian noise to add to parameters
        seed: Random seed for reproducibility

    Returns:
        Tuple of (perturbed_model, empty_losses_list)
    """
    # Copy the model
    model_copy = copy_model(policy._model, train_config)

    # Get current parameters
    params = nnx.state(model_copy)

    # Create RNG key for noise generation
    rng = jax.random.key(seed)

    # Get trainable parameters to count them
    trainable_filter = train_config.trainable_filter
    trainable_params = params.filter(trainable_filter)

    # Generate enough keys for all trainable parameters
    num_params = len(jax.tree_util.tree_leaves(trainable_params))
    keys = list(jax.random.split(rng, num_params))

    # Use a mutable counter to track which key to use
    key_idx = [0]

    def add_noise_to_param(param):
        """Add Gaussian noise to a single parameter."""
        if key_idx[0] >= len(keys):
            return param
        key = keys[key_idx[0]]
        key_idx[0] += 1

        if hasattr(param, "value"):
            # Handle nnx.Variable/Param types
            noise = jax.random.normal(key, param.value.shape, dtype=param.value.dtype) * noise_std
            return param.replace(param.value + noise)
        else:
            # Handle raw arrays
            noise = jax.random.normal(key, param.shape, dtype=param.dtype) * noise_std
            return param + noise

    # Apply noise to trainable parameters using state_map
    noisy_params = nnx_utils.state_map(
        params,
        trainable_filter,
        add_noise_to_param,
    )

    # Update the model with noisy parameters
    nnx.update(model_copy, noisy_params)

    pbar.write(f"[Gaussian Perturbation] Added noise (std={noise_std}) to {key_idx[0]} trainable parameters")

    # Return the perturbed model with empty losses (no training was performed)
    return model_copy, []


def _legacy_compute_alignment_ratio(
    policy: _policy.Policy,
    action_chunk: _model.Actions,
    curr_obs_dict: dict[str, Any],
    noise: np.ndarray | jnp.ndarray | None = None,
    cfg_weight: float = 1.0
) -> tuple[float, _model.Actions]:
    """
    Calculate the alignment ratio of the model.
    """

    original_prompt = curr_obs_dict["prompt"]
    # Use empty prompt
    if isinstance(original_prompt, list):
        curr_obs_dict["prompt"] = [""] * len(original_prompt)
    else:
        curr_obs_dict["prompt"] = ""
    if noise is not None and getattr(noise, "ndim", 0) == 3:
        action_chunks_empty = []
        for sample_idx in range(int(noise.shape[0])):
            action_i = policy.infer(curr_obs_dict, noise=noise[sample_idx])["actions"]
            action_i = action_i if isinstance(action_i, jax.Array) else jnp.asarray(action_i)
            action_chunks_empty.append(action_i)
        action_chunk_empty = jnp.stack(action_chunks_empty, axis=0)
    else:
        action_chunk_empty = policy.infer(curr_obs_dict, noise=noise)["actions"]
    curr_obs_dict["prompt"] = original_prompt

    action_chunk_jax = action_chunk if isinstance(action_chunk, jax.Array) else jnp.array(action_chunk)
    action_chunk_empty_jax = (
        action_chunk_empty if isinstance(action_chunk_empty, jax.Array) else jnp.array(action_chunk_empty)
    )

    if action_chunk_jax.ndim == 2:
        action_chunk_jax = action_chunk_jax[None, ...]
    if action_chunk_empty_jax.ndim == 2:
        action_chunk_empty_jax = action_chunk_empty_jax[None, ...]

    # Align to overlapping shape for robust ratio computation.
    s = min(action_chunk_jax.shape[0], action_chunk_empty_jax.shape[0])
    h = min(action_chunk_jax.shape[1], action_chunk_empty_jax.shape[1], 5)
    d = min(action_chunk_jax.shape[2], action_chunk_empty_jax.shape[2])
    action_chunk_overlap = action_chunk_jax[:s, :h, :d]
    action_chunk_empty_overlap = action_chunk_empty_jax[:s, :h, :d]

    # Compute ratio per sample and average for logging.
    num = jnp.linalg.norm(action_chunk_overlap - action_chunk_empty_overlap, axis=(1, 2))
    den = jnp.linalg.norm(action_chunk_empty_overlap, axis=(1, 2))
    alignment_ratio = float(jax.device_get(jnp.mean(num / den)))

    # Only apply CFG blend when action tensors match exactly.
    if action_chunk_jax.shape == action_chunk_empty_jax.shape:
        cfg_actions = action_chunk_empty + cfg_weight * (action_chunk - action_chunk_empty)
    else:
        cfg_actions = action_chunk
        print(
            "[TTT][warn] Skipping CFG blend due to action shape mismatch: "
            f"action={tuple(action_chunk_jax.shape)}, empty={tuple(action_chunk_empty_jax.shape)}"
        )

    return alignment_ratio, cfg_actions
