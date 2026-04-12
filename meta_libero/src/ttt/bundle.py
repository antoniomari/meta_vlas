## Training Function
# Suppress all warnings at the very beginning
import warnings

warnings.filterwarnings("ignore")
# Specifically suppress the JAX shape deprecation warning from Flax
warnings.filterwarnings(
    "ignore", message=".*shape requires ndarray or scalar arguments.*"
)
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*linear_util.wrap_init.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax.core.scope")

import dataclasses
import functools
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Sequence,
    SupportsIndex,
    Tuple,
    Union,
)
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

# Pair-batch layouts for logging per-stream losses (see train_step / train_model_on_fly).
PAIR_STREAM_NONE = 0
PAIR_STREAM_INTERLEAVED = 1  # PairDataset collate: s1,s2,s1,s2,... (task2 stream, task1 stream)
PAIR_STREAM_HALVES = 2  # First half vs second half (e.g. self-check: GT then pseudo)
import imageio
from PIL import Image, ImageDraw, ImageFont
import sys
import random
import re
import gc
import h5py

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops"

from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
import openpi.models.pi0_config as pi0_config
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# meta_libero/src/ttt/bundle.py -> parents[3] is repo (meta_vlas)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
from meta_libero.src.dataset import FilteredDataset, make_pseudo_label_inference_fn
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


def _duplicate_observation_along_batch(obs: _model.Observation) -> _model.Observation:
    """Concatenate observation with itself along batch dim (2*B samples)."""
    def cat_twice(x):
        if x is None:
            return None
        return jnp.concatenate([x, x], axis=0)

    return dataclasses.replace(
        obs,
        images={k: jnp.concatenate([v, v], axis=0) for k, v in obs.images.items()},
        image_masks={k: jnp.concatenate([v, v], axis=0) for k, v in obs.image_masks.items()},
        state=jnp.concatenate([obs.state, obs.state], axis=0),
        tokenized_prompt=cat_twice(obs.tokenized_prompt),
        tokenized_prompt_mask=cat_twice(obs.tokenized_prompt_mask),
        token_ar_mask=cat_twice(obs.token_ar_mask),
        token_loss_mask=cat_twice(obs.token_loss_mask),
    )


def _expand_self_check_batch(
    rng: at.KeyArrayLike,
    train_state: training_utils.TrainState,
    batch: Union[
        tuple[_model.Observation, _model.Actions],
        tuple[_model.Observation, _model.Actions, at.Array],
    ],
    lambda_kl: float,
) -> tuple[_model.Observation, _model.Actions, at.Array]:
    """Duplicate batch: first half ground-truth actions, second half pseudo-actions from current model."""
    observation, actions = batch[0], batch[1]
    model = nnx.merge(train_state.model_def, train_state.params)
    model.eval()
    step_i = int(jax.device_get(train_state.step))
    sample_rng = jax.random.fold_in(rng, step_i)
    pseudo_actions = jnp.asarray(
        model.sample_actions(sample_rng, observation), dtype=actions.dtype
    )
    obs_dup = _duplicate_observation_along_batch(observation)
    B = int(actions.shape[0])
    actions_dup = jnp.concatenate([actions, pseudo_actions], axis=0)
    base_w = (
        batch[2]
        if len(batch) == 3
        else jnp.ones(B, dtype=jnp.float32)
    )
    lam = jnp.asarray(lambda_kl, dtype=jnp.float32)
    weights = jnp.concatenate([base_w, base_w * lam])
    del model
    return (obs_dup, actions_dup, weights)


def _pair_stream_reduced_losses(
    loss_per_sample: jax.Array,
    sample_weights: jax.Array,
    pair_stream_layout: int,
) -> tuple[jax.Array, jax.Array]:
    """Mean loss on stream A vs B (weighted). layout=0 -> nan,nan."""
    nan = jnp.asarray(jnp.nan, dtype=loss_per_sample.dtype)
    if pair_stream_layout == PAIR_STREAM_NONE:
        return nan, nan
    wt = sample_weights.astype(loss_per_sample.dtype)
    B = loss_per_sample.shape[0]
    if pair_stream_layout == PAIR_STREAM_INTERLEAVED:
        m0 = (jnp.arange(B) % 2 == 0).astype(loss_per_sample.dtype)
        m1 = 1.0 - m0
    else:
        half = B // 2
        m0 = (jnp.arange(B) < half).astype(loss_per_sample.dtype)
        m1 = 1.0 - m0
    d0 = jnp.maximum(jnp.sum(wt * m0), 1e-8)
    d1 = jnp.maximum(jnp.sum(wt * m1), 1e-8)
    ls0 = jnp.sum(loss_per_sample * wt * m0) / d0
    ls1 = jnp.sum(loss_per_sample * wt * m1) / d1
    return ls0, ls1


@at.typecheck
@functools.partial(
    jax.jit,
    static_argnums=(0, 5, 6, 7, 8),
    donate_argnums=(2, 4),
)  # config, pair_stream_layout, l1_loss, kl_lambda, grpo_like static; ref_graphdef/ref_params dynamic
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: Union[
        tuple[_model.Observation, _model.Actions],
        tuple[_model.Observation, _model.Actions, at.Array],
    ],
    prev_grads: Any,
    pair_stream_layout: int = PAIR_STREAM_NONE,
    l1_loss: bool = False,
    kl_lambda: float = 0.0,
    grpo_like: bool = False,
    ref_graphdef: Any = None,
    ref_params: Any = None,
) -> tuple[training_utils.TrainState, dict[str, at.Array], Any]:
    # Memory checkpoint: start of train_step
    # NOTE: Removed all jax.lax.cond print statements to avoid recompilation
    # When step_int changes (0, 1, 2, then >= 3), JAX sees different execution paths
    # and recompiles the function, causing 42s delay on every step
    # Debug prints should be done outside the JIT function instead

    model = nnx.merge(state.model_def, state.params)
    model.train()

    ref_m: _model.BaseModel | None = None
    if float(kl_lambda) != 0.0 and ref_graphdef is not None:
        ref_m = nnx.merge(ref_graphdef, ref_params)
        ref_m.eval()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        sample_weights: at.Array,
    ):
        # Note: I set train=False to avoid preprocessing observation
        # TODO: consider what is the best thing to do here
        chunked_loss = model.compute_loss(
            rng,
            observation,
            actions,
            train=False,
            l1_loss=l1_loss,
            ref_model=ref_m,
            kl_lambda=kl_lambda,
        )
        # chunked_loss has shape (B, action_horizon); reduce to per-sample
        loss_per_sample = jnp.mean(chunked_loss, axis=tuple(range(1, chunked_loss.ndim)))
        if grpo_like:
            # Surrogate for sum_i A_i * log pi(a_i|s): mean_i A_i * denoise_loss_i (signed A_i in sample_weights).
            bsz = jnp.maximum(
                jnp.asarray(jnp.shape(actions)[0], dtype=loss_per_sample.dtype),
                jnp.asarray(1.0, dtype=loss_per_sample.dtype),
            )
            return jnp.sum(loss_per_sample * sample_weights) / bsz
        # Weighted loss: sum(loss * weights) / max(sum(weights), eps); weights=1 gives mean loss
        return jnp.sum(loss_per_sample * sample_weights) / jnp.maximum(
            jnp.sum(sample_weights), 1e-8
        )

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch[0], batch[1]
    batch_size = jnp.shape(actions)[0]
    sample_weights = batch[2] if len(batch) == 3 else jnp.ones(batch_size)

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions, sample_weights
    )

    if pair_stream_layout != PAIR_STREAM_NONE:
        chunked_m = model.compute_loss(
            train_rng,
            observation,
            actions,
            train=False,
            l1_loss=l1_loss,
            ref_model=ref_m,
            kl_lambda=kl_lambda,
        )
        lps_m = jnp.mean(chunked_m, axis=tuple(range(1, chunked_m.ndim)))
        ls0, ls1 = _pair_stream_reduced_losses(
            lps_m, sample_weights, pair_stream_layout
        )
    else:
        ls0 = ls1 = jnp.asarray(jnp.nan, dtype=loss.dtype)

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

    # Magnitude of parameter shift: L2 norm of the updates applied this step.
    update_norm = optax.global_norm(updates)

    # Cosine similarity between current and previous gradient (nan at step 0).
    grad_norm_curr = optax.global_norm(grads)
    grad_norm_prev = optax.global_norm(prev_grads)
    grad_dot = sum(
        jnp.sum(g * p)
        for g, p in zip(
            jax.tree_util.tree_leaves(grads),
            jax.tree_util.tree_leaves(prev_grads),
        )
    )
    grad_cosine_sim = jnp.where(
        grad_norm_prev > 1e-9,
        grad_dot / (grad_norm_curr * grad_norm_prev + 1e-12),
        jnp.nan,
    )

    info = {
        "loss": loss,
        "grad_norm": grad_norm_curr,
        "param_norm": optax.global_norm(kernel_params),
        "update_norm": update_norm,
        "grad_cosine_sim": grad_cosine_sim,
    }
    if pair_stream_layout != PAIR_STREAM_NONE:
        info["loss_stream_0"] = ls0
        info["loss_stream_1"] = ls1

    return new_state, info, grads


@functools.partial(
    jax.jit,
    static_argnums=(0, 4, 5, 6, 7),
    # Do not donate `state`: gradient accumulation calls this multiple times per optimizer
    # step with the same train_state (trajectory microbatches or accum_steps>1). Donating
    # state on the first call invalidates buffers and can trigger XLA errors (e.g.
    # PjRtBuffer::layout / has_layout) on subsequent calls.
    donate_argnums=(),
)
def _compute_grads_and_loss(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: Union[
        tuple[_model.Observation, _model.Actions],
        tuple[_model.Observation, _model.Actions, at.Array],
    ],
    pair_stream_layout: int = PAIR_STREAM_NONE,
    l1_loss: bool = False,
    kl_lambda: float = 0.0,
    grpo_like: bool = False,
    ref_graphdef: Any = None,
    ref_params: Any = None,
) -> tuple[dict, at.Array, jax.Array, jax.Array]:
    """Compute gradients and loss for a batch without applying updates. For gradient accumulation."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    ref_m: _model.BaseModel | None = None
    if float(kl_lambda) != 0.0 and ref_graphdef is not None:
        ref_m = nnx.merge(ref_graphdef, ref_params)
        ref_m.eval()

    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        sample_weights: at.Array,
    ):
        chunked_loss = model.compute_loss(
            rng,
            observation,
            actions,
            train=False,
            l1_loss=l1_loss,
            ref_model=ref_m,
            kl_lambda=kl_lambda,
        )
        loss_per_sample = jnp.mean(chunked_loss, axis=tuple(range(1, chunked_loss.ndim)))
        if grpo_like:
            bsz = jnp.maximum(
                jnp.asarray(jnp.shape(actions)[0], dtype=loss_per_sample.dtype),
                jnp.asarray(1.0, dtype=loss_per_sample.dtype),
            )
            return jnp.sum(loss_per_sample * sample_weights) / bsz
        return jnp.sum(loss_per_sample * sample_weights) / jnp.maximum(
            jnp.sum(sample_weights), 1e-8
        )

    # Use provided rng directly (caller folds in step+accum_i for per-micro-batch randomness)
    train_rng = rng
    observation, actions = batch[0], batch[1]
    batch_size = jnp.shape(actions)[0]
    sample_weights = batch[2] if len(batch) == 3 else jnp.ones(batch_size)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions, sample_weights
    )
    if pair_stream_layout != PAIR_STREAM_NONE:
        chunked_m = model.compute_loss(
            train_rng,
            observation,
            actions,
            train=False,
            l1_loss=l1_loss,
            ref_model=ref_m,
            kl_lambda=kl_lambda,
        )
        lps_m = jnp.mean(chunked_m, axis=tuple(range(1, chunked_m.ndim)))
        ls0, ls1 = _pair_stream_reduced_losses(
            lps_m, sample_weights, pair_stream_layout
        )
    else:
        ls0 = ls1 = jnp.asarray(jnp.nan, dtype=loss.dtype)
    return grads, loss, ls0, ls1


@functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(2, 3))
def _apply_accumulated_grads(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    accumulated_grads: dict,
    prev_grads: Any,
) -> tuple[training_utils.TrainState, dict[str, at.Array], Any]:
    """Apply averaged accumulated gradients and return new state. For gradient accumulation."""
    model = nnx.merge(state.model_def, state.params)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(
        accumulated_grads, state.opt_state, params
    )
    new_params = optax.apply_updates(params, updates)
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
    update_norm = optax.global_norm(updates)
    grad_norm_curr = optax.global_norm(accumulated_grads)
    grad_norm_prev = optax.global_norm(prev_grads)
    grad_dot = sum(
        jnp.sum(g * p)
        for g, p in zip(
            jax.tree_util.tree_leaves(accumulated_grads),
            jax.tree_util.tree_leaves(prev_grads),
        )
    )
    grad_cosine_sim = jnp.where(
        grad_norm_prev > 1e-9,
        grad_dot / (grad_norm_curr * grad_norm_prev + 1e-12),
        jnp.nan,
    )
    info = {
        "grad_norm": grad_norm_curr,
        "param_norm": optax.global_norm(kernel_params),
        "update_norm": update_norm,
        "grad_cosine_sim": grad_cosine_sim,
    }
    return new_state, info, accumulated_grads


# Matches OpenPI Pi0 action-expert params (see pi0_config.get_freeze_filter / PathRegex(".*llm.*_1.*")).
_ACTION_EXPERT_PATH_RE = re.compile(r".*llm.*_1.*")


def _lr_schedule_for_peak(learning_rate: float, warmup_steps: int):
    """Same schedule shape as init_train_state: warmup then constant, or cosine decay when no warmup."""
    if warmup_steps > 0:
        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, learning_rate, warmup_steps),
                optax.constant_schedule(learning_rate),
            ],
            boundaries=[warmup_steps],
        )
    return optax.warmup_cosine_decay_schedule(
        init_value=learning_rate,
        peak_value=learning_rate,
        warmup_steps=0,
        decay_steps=30000,
        end_value=learning_rate * 0.1,
    )


def _optimizer_branch_no_wd(lr_peak: float, warmup_steps: int) -> optax.GradientTransformation:
    return optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(_lr_schedule_for_peak(lr_peak, warmup_steps)),
        optax.scale(-1.0),
    )


def _optimizer_branch_wd(
    weight_decay: float, lr_peak: float, warmup_steps: int
) -> optax.GradientTransformation:
    return optax.chain(
        optax.scale_by_adam(),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_schedule(_lr_schedule_for_peak(lr_peak, warmup_steps)),
        optax.scale(-1.0),
    )


def _param_labels_action_expert_vs_base(trainable_params: Any) -> Any:
    """PyTree matching trainable_params; each leaf is 'action_expert' or 'base' for multi_transform."""

    def label(path: Any, _leaf: Any) -> str:
        path_s = jax.tree_util.keystr(path)
        return "action_expert" if _ACTION_EXPERT_PATH_RE.search(path_s) else "base"

    return jax.tree_util.tree_map_with_path(label, trainable_params)


def init_train_state_two_lrs(
    config: _config.TrainConfig,
    weight_decay: float,
    warmup_steps: int,
    lr_action_expert: float,
    lr_base: float,
    graphdef: nnx.GraphDef,
    params: at.Params,
) -> training_utils.TrainState:
    """Train state with separate Adam + LR schedules for action expert vs rest (not JIT — label tree is Python)."""
    trainable = params.filter(config.trainable_filter)
    labels = _param_labels_action_expert_vs_base(trainable)
    if weight_decay > 0:
        transforms = {
            "action_expert": _optimizer_branch_wd(
                weight_decay, lr_action_expert, warmup_steps
            ),
            "base": _optimizer_branch_wd(weight_decay, lr_base, warmup_steps),
        }
    else:
        transforms = {
            "action_expert": _optimizer_branch_no_wd(lr_action_expert, warmup_steps),
            "base": _optimizer_branch_no_wd(lr_base, warmup_steps),
        }
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(transforms, labels),
    )
    opt_state = tx.init(trainable)
    return training_utils.TrainState(
        step=0,
        params=params,
        model_def=graphdef,
        tx=tx,
        opt_state=opt_state,
        ema_decay=config.ema_decay,
        ema_params=None if config.ema_decay is None else params,
    )


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

    lr_schedule = _lr_schedule_for_peak(learning_rate, warmup_steps)

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


def _compute_validation_loss(
    model: _model.BaseModel,
    validation_batches: list[tuple[_model.Observation, _model.Actions]],
    seed: int,
) -> float:
    """Compute mean loss over cached validation batches (no gradients, no disk I/O)."""
    model.eval()
    total_loss = 0.0
    rng = jax.random.key(seed)
    for observation, actions in validation_batches:
        rng, step_rng = jax.random.split(rng)
        chunked_loss = model.compute_loss(step_rng, observation, actions, train=False)
        total_loss += float(jax.device_get(jnp.mean(chunked_loss)))
    return total_loss / len(validation_batches) if validation_batches else 0.0


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


def _index_tree_at(tree: Any, idx: int) -> Any:
    """Index a PyTree at the given batch index (for arrays)."""

    def _index(x: Any) -> Any:
        if hasattr(x, "shape") and len(getattr(x, "shape", ())) > 0:
            arr = np.asarray(x) if hasattr(x, "numpy") or hasattr(x, "__array__") else x
            return arr[idx]
        return x

    return jax.tree.map(_index, tree)


def _compute_alignment_weights_for_batch(
    policy: Any,
    observation: _model.Observation,
    actions: at.Array,
    threshold: float,
    rng: at.KeyArrayLike,
    task1_only: bool,
) -> tuple[jnp.ndarray, float]:
    """Compute sample weights: 1 if alignment_ratio <= threshold else 0.

    When task1_only=True, only task1 samples (odd indices) are filtered; task2 get weight=1.
    Returns (weights, avg_ratio) where avg_ratio is the mean alignment ratio over task1 samples.
    """
    batch_size = int(np.shape(actions)[0])
    weights = np.ones(batch_size, dtype=np.float32)
    if not task1_only:
        return jnp.asarray(weights), 0.0
    task1_indices = list(range(1, batch_size, 2))
    if not task1_indices:
        return jnp.asarray(weights), 0.0
    obs_dict = observation.to_dict() if hasattr(observation, "to_dict") else observation
    if "prompt" not in obs_dict:
        obs_dict = dict(obs_dict)
        obs_dict["prompt"] = ""
    horizon, action_dim = int(actions.shape[1]), int(actions.shape[2])
    ratio_sum = 0.0
    for i, idx in enumerate(task1_indices):
        obs_i = _index_tree_at(obs_dict, idx)
        obs_i = dict(obs_i) if isinstance(obs_i, dict) else obs_i
        if isinstance(obs_i, dict) and "prompt" not in obs_i:
            obs_i = dict(obs_i)
            obs_i["prompt"] = ""
        actions_i = np.asarray(actions[idx])
        if actions_i.ndim == 2:
            actions_i = actions_i[None, ...]
        step_rng = jax.random.fold_in(rng, idx)
        noise = jax.random.normal(step_rng, (1, horizon, action_dim))
        try:
            ratios = compute_alignment_ratio(
                policy, actions_i, obs_i, noise=noise, return_per_sample=True
            )
            ratios_arr = jnp.asarray(ratios)
            ratio_val = float(
                jax.device_get(ratios_arr[0] if ratios_arr.ndim > 0 else ratios_arr)
            )
        except (KeyError, IndexError, TypeError):
            ratio_val = 0.0
        ratio_sum += ratio_val
        weights[idx] = 1.0 if ratio_val <= threshold else 0.0
    avg_ratio = ratio_sum / len(task1_indices)
    return jnp.asarray(weights), avg_ratio


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
    show_progress_bar: bool = True,
    # Resume parameters - pass these to continue training from a previous run
    resume_train_state: training_utils.TrainState | None = None,
    resume_losses: list[float] | None = None,
    # Control buffer donation - disable for TTT since we extract model immediately after
    donate_buffers: bool = True,
    # Optional auxiliary denoising-loss evaluation on each optimization step.
    aux_observation: _model.Observation | None = None,
    aux_actions: _model.Actions | None = None,
    aux_num_samples: int = 10,
    # Optional callback (step, loss) after each gradient step (e.g. for UI streaming).
    on_step_callback: Optional[Callable[[int, float], None]] = None,
    # Optional callback (step, info_dict) with full step info (loss, grad_norm, update_norm).
    on_step_info_callback: Optional[Callable[[int, dict], None]] = None,
    # Optional validation: compute loss on validation set every N steps.
    validation_data_loader: Optional[_data_loader.DataLoader] = None,
    validation_interval: int = 5,
    on_validation_callback: Optional[Callable[[int, float], None]] = None,
    # Optional evaluation: run env evaluation every N steps (e.g. every 20 steps).
    evaluation_interval: int = 20,
    on_evaluation_callback: Optional[Callable[[int, _model.BaseModel], None]] = None,
    # Self-replay alignment weighting: filter task1 samples by alignment ratio.
    alignment_ratio_threshold: float | None = None,
    alignment_reference_policy: Optional[Any] = None,
    alignment_weight_task1_only: bool = False,
    # Gradient accumulation: accumulate gradients over N batches before applying (e.g. 2 when batch size is halved).
    gradient_accumulation_steps: int = 1,
    # Dynamic self-check: duplicate each batch with pseudo-actions from the current training model (not a fixed ref).
    self_check_lambda_kl: float | None = None,
    # Two learning rates: action expert (Pi0 llm *_1*) vs rest; uses optax.multi_transform (init not JIT).
    two_lrs: bool = False,
    lr_action_expert: float = 2.5e-4,
    lr_base: float = 2.5e-5,
    # Log per-stream mean loss for paired batches (PAIR_STREAM_INTERLEAVED / PAIR_STREAM_HALVES).
    pair_stream_layout: int = PAIR_STREAM_NONE,
    # Pi0 diffusion BC: L1 on residual (|v-u|) instead of L2 (squared error).
    l1_loss: bool = False,
    # Pi0 only: extra MSE between student v_t and stop-grad(ref) v_t (same noise/time/x_t); weight kl_lambda.
    kl_lambda: float = 0.0,
    ref_model_for_kl: _model.BaseModel | None = None,
    # GRPO-like: batch loss = mean_i (advantage_i * denoise_loss_i); advantages in distill_sample_weight (signed).
    grpo_like: bool = False,
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
        if two_lrs:
            train_state = init_train_state_two_lrs(
                config,
                weight_decay,
                warmup_steps,
                lr_action_expert,
                lr_base,
                graphdef,
                initial_params,
            )
        else:
            train_state = init_train_state(
                config, weight_decay, learning_rate, warmup_steps, graphdef, initial_params
            )
        # NOTE: after this we go from 12.58 GB -> 25.13 GB why?
        print_memory_checkpoint("After JIT compilation of init_train_state", 566)

    # Initialize RNG
    rng = jax.random.key(seed)
    # Define trainable_filter for training step
    _trainable_filter = _trainable_filter_for_init

    if grpo_like:
        if int(gradient_accumulation_steps) != 1:
            raise ValueError(
                "train_model_on_fly: grpo_like is incompatible with gradient_accumulation_steps != 1"
            )
        if not hasattr(training_data_loader, "iter_global_microbatches"):
            raise ValueError(
                "train_model_on_fly: grpo_like requires a data loader with iter_global_microbatches() "
                "(e.g. NeighborsDataLoader with grpo_full_buffer=True)."
            )

    if float(kl_lambda) != 0.0 and ref_model_for_kl is None:
        raise ValueError(
            "train_model_on_fly: kl_lambda != 0 requires ref_model_for_kl (frozen snapshot of the student)."
        )
    kl_ref_graphdef: Any = None
    kl_ref_params: Any = None
    if float(kl_lambda) != 0.0:
        kl_ref_graphdef = nnx.graphdef(ref_model_for_kl)
        kl_ref_params = nnx.state(ref_model_for_kl)

    # train_step is now automatically JIT-compiled via decorator
    # No need to create train_step_jit - just call train_step directly
    # JAX will automatically cache the compilation based on function signature and input shapes
    train_step_jit = functools.partial(
        train_step,
        config,
        pair_stream_layout=pair_stream_layout,
        l1_loss=l1_loss,
        kl_lambda=float(kl_lambda),
        grpo_like=grpo_like,
        ref_graphdef=kl_ref_graphdef,
        ref_params=kl_ref_params,
    )
    compute_grads_loss_jit = functools.partial(
        _compute_grads_and_loss,
        config,
        pair_stream_layout=pair_stream_layout,
        l1_loss=l1_loss,
        kl_lambda=float(kl_lambda),
        grpo_like=grpo_like,
        ref_graphdef=kl_ref_graphdef,
        ref_params=kl_ref_params,
    )

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
        disable=not show_progress_bar,
    )

    # Initialize iterator
    data_iter = iter(training_data_loader)

    # Cache validation batches once to avoid re-loading from disk every validation step
    validation_batches: list[tuple[_model.Observation, _model.Actions]] | None = None
    if validation_data_loader is not None:
        validation_batches = list(validation_data_loader)
        if not validation_batches:
            validation_batches = None

    # Following marco.py: no separate warmup, first training step will compile naturally
    print_memory_checkpoint("Before training loop starts", 604)

    # Zero-initialized prev_grads for step 0 (grad_cosine_sim will be nan)
    trainable_params = train_state.params.filter(config.trainable_filter)

    def _zero_like(x):
        v = x.value if hasattr(x, "value") else x
        return jnp.zeros_like(v)

    prev_grads = jax.tree.map(_zero_like, trainable_params)
    del trainable_params

    accum_steps = gradient_accumulation_steps
    if accum_steps > 1:
        print(f"Gradient accumulation: {accum_steps} steps per optimizer update")
    if self_check_lambda_kl is not None:
        print(
            f"Dynamic self-check: pseudo-labels from current model each step, lambda_kl={self_check_lambda_kl}"
        )
    if l1_loss:
        print("BC loss: L1 (mean |v_t - u_t| on diffusion residual); default is L2 MSE.")
    if float(kl_lambda) != 0.0:
        print(
            f"KL proxy: MSE(student v_t vs frozen ref v_t, same noise) weighted by kl_lambda={kl_lambda}"
        )
    if grpo_like:
        print(
            "GRPO-like objective: minimize mean_i (advantage_i * diffusion_BC_loss_i); "
            "advantages (signed) must be stored in batch distill_sample_weight."
        )
    if grpo_like:
        print(
            "GRPO: one optimizer update per --n_epoch pass over the pooled buffer (all rollouts); "
            "microbatch gradients are averaged (same policy state across microbatches)."
        )

    traj_gen = None
    if grpo_like:
        traj_gen = training_data_loader.iter_global_microbatches()
    data_iter = iter(training_data_loader) if traj_gen is None else None

    for step in pbar:
        t0_train = time.perf_counter()
        t0_fetch = time.perf_counter()

        if traj_gen is not None:
            try:
                traj_batches = next(traj_gen)
            except StopIteration as e:
                raise RuntimeError(
                    "GRPO pooled-buffer microbatch iterator exhausted before completing num_steps"
                ) from e
            t1_fetch = time.perf_counter()
            batch_fetch_time = (t1_fetch - t0_fetch) * 1000
            accum_grads = None
            accum_losses = []
            accum_ls0 = []
            accum_ls1 = []
            alignment_ratios = []
            n_micro = len(traj_batches)
            if n_micro < 1:
                raise RuntimeError("trajectory produced zero microbatches")
            for accum_i in range(n_micro):
                batch = traj_batches[accum_i]
                if self_check_lambda_kl is not None:
                    batch = _expand_self_check_batch(
                        rng, train_state, batch, self_check_lambda_kl
                    )
                if (
                    alignment_ratio_threshold is not None
                    and alignment_reference_policy is not None
                    and alignment_weight_task1_only
                ):
                    step_rng = jax.random.fold_in(
                        rng, int(step) * 100_000 + accum_i
                    )
                    sample_weights, avg_ratio = _compute_alignment_weights_for_batch(
                        policy=alignment_reference_policy,
                        observation=batch[0],
                        actions=batch[1],
                        threshold=alignment_ratio_threshold,
                        rng=step_rng,
                        task1_only=True,
                    )
                    if len(batch) == 3:
                        sample_weights = sample_weights * batch[2]
                    batch = (batch[0], batch[1], sample_weights)
                    alignment_ratios.append(avg_ratio)
                micro_rng = jax.random.fold_in(
                    rng, int(step) * 100_000 + accum_i
                )
                grads, loss, ls0_m, ls1_m = compute_grads_loss_jit(
                    micro_rng, train_state, batch
                )
                if pair_stream_layout != PAIR_STREAM_NONE:
                    accum_ls0.append(ls0_m)
                    accum_ls1.append(ls1_m)
                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = jax.tree.map(
                        lambda a, b: a + b, accum_grads, grads
                    )
                    del grads
                accum_losses.append(loss)
            mean_grads = jax.tree.map(
                lambda g: g / float(n_micro), accum_grads
            )
            del accum_grads
            train_state, info, prev_grads = _apply_accumulated_grads(
                config, train_state, mean_grads, prev_grads
            )
            del mean_grads
            mean_loss = jnp.mean(jnp.stack(accum_losses))
            info["loss"] = mean_loss
            if pair_stream_layout != PAIR_STREAM_NONE and accum_ls0:
                info["loss_stream_0"] = jnp.mean(jnp.stack(accum_ls0))
                info["loss_stream_1"] = jnp.mean(jnp.stack(accum_ls1))
            alignment_step_info = (
                {"alignment_task1_avg_ratio": float(np.mean(alignment_ratios))}
                if alignment_ratios
                else {}
            )
        elif accum_steps == 1:
            # Single-step path: fetch one batch, apply train_step
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(training_data_loader)
                batch = next(data_iter)
            t1_fetch = time.perf_counter()
            batch_fetch_time = (t1_fetch - t0_fetch) * 1000
            if PRINT_TIME_EXECUTION:
                print(f"[TIMING] Batch fetch took {batch_fetch_time:.2f} ms")

            if self_check_lambda_kl is not None:
                batch = _expand_self_check_batch(
                    rng, train_state, batch, self_check_lambda_kl
                )

            if (
                alignment_ratio_threshold is not None
                and alignment_reference_policy is not None
                and alignment_weight_task1_only
            ):
                step_rng = jax.random.fold_in(rng, step)
                sample_weights, avg_ratio = _compute_alignment_weights_for_batch(
                    policy=alignment_reference_policy,
                    observation=batch[0],
                    actions=batch[1],
                    threshold=alignment_ratio_threshold,
                    rng=step_rng,
                    task1_only=True,
                )
                if len(batch) == 3:
                    sample_weights = sample_weights * batch[2]
                batch = (batch[0], batch[1], sample_weights)
                alignment_step_info = {"alignment_task1_avg_ratio": avg_ratio}
            else:
                alignment_step_info = {}

            train_state, info, prev_grads = train_step_jit(
                rng, train_state, batch, prev_grads
            )
        else:
            # Gradient accumulation: fetch N batches, compute grads, average, apply
            accum_grads = None
            accum_losses = []
            accum_ls0: list[jax.Array] = []
            accum_ls1: list[jax.Array] = []
            alignment_ratios = []
            for accum_i in range(accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(training_data_loader)
                    batch = next(data_iter)
                if self_check_lambda_kl is not None:
                    batch = _expand_self_check_batch(
                        rng, train_state, batch, self_check_lambda_kl
                    )
                if (
                    alignment_ratio_threshold is not None
                    and alignment_reference_policy is not None
                    and alignment_weight_task1_only
                ):
                    step_rng = jax.random.fold_in(
                        rng, step * accum_steps + accum_i
                    )
                    sample_weights, avg_ratio = _compute_alignment_weights_for_batch(
                        policy=alignment_reference_policy,
                        observation=batch[0],
                        actions=batch[1],
                        threshold=alignment_ratio_threshold,
                        rng=step_rng,
                        task1_only=True,
                    )
                    if len(batch) == 3:
                        sample_weights = sample_weights * batch[2]
                    batch = (batch[0], batch[1], sample_weights)
                    alignment_ratios.append(avg_ratio)
                micro_rng = jax.random.fold_in(
                    rng, int(step) * accum_steps + accum_i
                )
                grads, loss, ls0_m, ls1_m = compute_grads_loss_jit(
                    micro_rng, train_state, batch
                )
                if pair_stream_layout != PAIR_STREAM_NONE:
                    accum_ls0.append(ls0_m)
                    accum_ls1.append(ls1_m)
                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = jax.tree.map(
                        lambda a, b: a + b, accum_grads, grads
                    )
                    del grads  # Free promptly to reduce peak memory
                accum_losses.append(loss)
            t1_fetch = time.perf_counter()
            batch_fetch_time = (t1_fetch - t0_fetch) * 1000

            mean_grads = jax.tree.map(
                lambda g: g / accum_steps, accum_grads
            )
            del accum_grads  # Free before apply to reduce peak memory
            train_state, info, prev_grads = _apply_accumulated_grads(
                config, train_state, mean_grads, prev_grads
            )
            del mean_grads
            mean_loss = jnp.mean(jnp.stack(accum_losses))
            info["loss"] = mean_loss
            if pair_stream_layout != PAIR_STREAM_NONE and accum_ls0:
                info["loss_stream_0"] = jnp.mean(jnp.stack(accum_ls0))
                info["loss_stream_1"] = jnp.mean(jnp.stack(accum_ls1))
            alignment_step_info = (
                {"alignment_task1_avg_ratio": float(np.mean(alignment_ratios))}
                if alignment_ratios
                else {}
            )

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
        if pair_stream_layout != PAIR_STREAM_NONE and "loss_stream_0" in info:
            jax.block_until_ready(info["loss_stream_0"])
            jax.block_until_ready(info["loss_stream_1"])
        t1_block = time.perf_counter()
        block_time = (t1_block - t0_block) * 1000  # Time to block until ready

        t1_train = time.perf_counter()
        train_step_time = (t1_train - t0_train) * 1000  # Total time
        time_before_call = 0.0  # t0_train is at loop start

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
        update_norm = float(jax.device_get(info["update_norm"]))
        grad_cosine_sim = float(jax.device_get(info["grad_cosine_sim"]))
        losses.append(loss_val)  # Store loss for plotting
        if on_step_callback is not None:
            on_step_callback(step, loss_val)
        if on_step_info_callback is not None:
            step_info = {
                "loss": loss_val,
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "grad_cosine_sim": grad_cosine_sim,
            }
            step_info.update(alignment_step_info)
            if pair_stream_layout != PAIR_STREAM_NONE and "loss_stream_0" in info:
                lv0 = float(jax.device_get(info["loss_stream_0"]))
                lv1 = float(jax.device_get(info["loss_stream_1"]))
                if not math.isnan(lv0) and not math.isnan(lv1):
                    step_info["loss_stream_0"] = lv0
                    step_info["loss_stream_1"] = lv1
            on_step_info_callback(step, step_info)

        # Optional: compute validation loss at step 0 and every N steps (uses cached batches, no disk I/O).
        if (
            validation_batches is not None
            and on_validation_callback is not None
            and step % validation_interval == 0
        ):
            val_model = nnx.merge(train_state.model_def, train_state.params)
            val_model.eval()
            val_loss = _compute_validation_loss(val_model, validation_batches, seed)
            on_validation_callback(step, val_loss)
            del val_model
            gc.collect()  # Free Python refs promptly; avoids OOM on next step with grad accumulation

        # Optional: run env evaluation every N steps (e.g. every 20 steps).
        if (
            on_evaluation_callback is not None
            and step % evaluation_interval == 0
        ):
            eval_model = nnx.merge(train_state.model_def, train_state.params)
            eval_model.eval()
            on_evaluation_callback(step, eval_model)
            del eval_model
            gc.collect()

        # Optional: compute auxiliary denoising loss after each optimization step.
        if aux_enabled:
            step_model = nnx.merge(train_state.model_def, train_state.params)
            step_model.eval()
            step_aux_losses = _compute_aux_denoising_losses(
                model=step_model,
                observation=batch[0],
                actions=batch[1],
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
            f"Step {step}: train={train_step_time:6.2f}ms, loss={loss_val:.4f} grad_norm={grad_norm:.4f}"
        )

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
    teacher_policy: _policy.Policy | None = None,
    show_progress_bar: bool = True,
    distillation_examples_out: list[dict[str, Any]] | None = None,
    student_action_merge: float = 0.0,
    group_size: int = 1,
    temporal_decay: float = 1.0,
    teacher_group_size: int = 1,
    write_auxiliary_rollout_pdfs: bool = True,
    after_each_rollout_episode: Callable[[int, int, int], None] | None = None,
    grpo_like: bool = False,
    grpo_trust_eps: float | None = None,
    grpo_weight: str | None = None,
    grpo_weight_eps: float = 1e-8,
    distill_collect_every: int = 1,
    distill_trajectory_id_offset: int = 0,
    num_envs: int = 1,
):
    """Run evaluation on a LIBERO task.

    Args:
        policy: The policy to evaluate. For reproducible results, create the policy with
                `create_policy(..., rng_seed=seed)` to initialize its internal RNG state.
        teacher_policy: If set, after each replanning inference of ``policy`` (same obs and
                noise), runs the teacher and records mean L2 distance between action chunks;
                saves ``teacher_student_action_l2_task{task_id}.pdf`` under ``video_out_path``.
        distillation_examples_out: If set (and ``teacher_policy`` is set), on-policy BC rows are
                appended when ``(t - num_steps_wait) % distill_collect_every == 0`` (``t`` = env
                step; default ``distill_collect_every=1`` matches every step after wait). The env
                still advances with ``replan_steps=5`` open-loop chunks. Without a teacher, no
                distillation rows.
        student_action_merge: In ``[0, 1]``. BC target actions are
                ``(1 - α) * teacher_chunk + α * student_chunk`` with ``α = student_action_merge``.
                ``0.0`` (default) matches pure teacher targets; ``1.0`` is pure student.
        group_size: Number of independent noise samples per replan (student). ``1`` matches the
                previous single-sample rollout. ``> 1`` runs G student forwards; the first sample
                drives the env; if ``teacher_policy`` is set and ``write_auxiliary_rollout_pdfs``,
                logs/plots per-episode ``group_action_sampling_ep*.pdf``.
        teacher_group_size: Independent teacher noise samples per replan (for variance / alignment);
                first sample is used for BC targets and teacher–student L2. ``1`` matches single teacher.
        write_auxiliary_rollout_pdfs: If False, skip standalone rollout PDFs (group sampling, teacher–student
                L2, per-episode alignment popups); callers can build a single combined PDF from
                ``run_evaluation.last_episode_metrics``.
        show_progress_bar: Episode tqdm bar. Set False when calling from inside another tqdm
                (e.g. mid-training eval); otherwise both bars fight for the terminal and log files
                fill with interleaved ANSI cursor spam.
        num_trials: Number of evaluation episodes to run.
        num_envs: Parallel LIBERO environments for distillation only; >1 batches policy inference
            at replan (requires teacher distillation and ``disable_adaptation=True`` in the TTT runner).
        task_suite_name: Name of the LIBERO task suite (e.g., "libero_10", "libero_90").
        task_id: ID of the task within the suite to evaluate.
        num_steps_wait: Number of steps to wait for environment stabilization.
        save_video: Whether to save rollout videos.
        video_out_path: Directory path to save videos.
        task_description: Description of the task (overridden by actual task description).
        seed: Random seed for environment and other randomness (but NOT policy RNG -
              policy RNG must be set during policy creation via create_policy's rng_seed parameter).
        after_each_rollout_episode: Optional callback after each episode:
              ``(episode_idx, task_episodes, task_successes)``.

    Returns:
        success_rate: Success rate across all evaluation episodes.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    if teacher_group_size < 1:
        raise ValueError(f"teacher_group_size must be >= 1, got {teacher_group_size}")
    if temporal_decay < 0:
        raise ValueError(f"temporal_decay must be >= 0, got {temporal_decay}")
    if grpo_like and int(group_size) < 2:
        raise ValueError(
            f"--grpo_like requires group_size >= 2, got {group_size}"
        )
    if grpo_trust_eps is not None and not grpo_like:
        raise ValueError("grpo_trust_eps requires grpo_like=True")
    if grpo_trust_eps is not None and float(grpo_trust_eps) <= 0.0:
        raise ValueError(f"grpo_trust_eps must be > 0 when set, got {grpo_trust_eps}")
    _gw = grpo_weight if grpo_weight is not None else "none"
    if _gw not in ("none", "mean_std"):
        raise ValueError(
            f"grpo_weight must be 'none' or 'mean_std', got {grpo_weight!r}"
        )
    if _gw == "mean_std" and not grpo_like:
        raise ValueError("grpo_weight='mean_std' requires grpo_like=True")
    if float(grpo_weight_eps) <= 0.0:
        raise ValueError(f"grpo_weight_eps must be > 0, got {grpo_weight_eps}")
    if int(distill_collect_every) < 1:
        raise ValueError(
            f"distill_collect_every must be >= 1, got {distill_collect_every}"
        )
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
        teacher_policy=teacher_policy,
        show_progress_bar=show_progress_bar,
        distillation_examples_out=distillation_examples_out,
        student_action_merge=student_action_merge,
        num_samples=group_size,
        temporal_decay=temporal_decay,
        teacher_group_size=teacher_group_size,
        write_auxiliary_rollout_pdfs=write_auxiliary_rollout_pdfs,
        after_each_rollout_episode=after_each_rollout_episode,
        grpo_like=grpo_like,
        grpo_trust_eps=grpo_trust_eps,
        grpo_weight=grpo_weight,
        grpo_weight_eps=grpo_weight_eps,
        distill_collect_every=distill_collect_every,
        distill_trajectory_id_offset=distill_trajectory_id_offset,
        num_envs=num_envs,
    )


# Create a simple dataloader that returns the same batch every time
class NeighborsDataLoader:
    """A simple dataloader that returns the same batch (obs, actions) every time.

    **Memory:** The full collated buffer is kept on **CPU** (NumPy). Only each minibatch
    is converted to device arrays in ``__iter__``. Holding the entire distillation /
    GRPO buffer on GPU (the previous behavior) scales with ``N ×`` image size and
    routinely OOMs when GRPO multiplies rows by ``group_size``.

    **Modes:**

    * **Default (``sequential_epochs=False``):** infinite iterator with **random** minibatches
      **with replacement** (legacy BC / distillation).
    * **Sequential epochs (``sequential_epochs=True``):** each call to ``iter()`` yields
      one **shuffled** pass over the dataset (non-overlapping batches of ``batch_size``,
      last batch may be smaller), then **StopIteration**. Matches RL-style GRPO/PPO-style
      reuse: ``num_steps = n_epoch * len(loader)`` in ``train_model_on_fly`` replays the
      iterator after each epoch.
    * **GRPO pooled buffer (``grpo_full_buffer=True``):** all rollout rows are concatenated;
      ``iter_global_microbatches()`` yields one **list** of microbatches per pass that covers **every**
      row (shuffled, split by ``batch_size``); ``train_model_on_fly(..., grpo_like=True)`` performs one
      optimizer update per yield, averaging gradients over that list. Incompatible with
      ``sequential_epochs=True``.
    """

    def __init__(
        self,
        examples: List[Any],
        batch_size: int,
        data_config: _config.DataConfig,
        *,
        sequential_epochs: bool = False,
        epoch_seed: int = 0,
        n_epoch: int = 1,
        grpo_full_buffer: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if sequential_epochs and grpo_full_buffer:
            raise ValueError(
                "NeighborsDataLoader: sequential_epochs and grpo_full_buffer are mutually exclusive"
            )
        self.batch_size = batch_size
        self._data_config = data_config
        self._sequential_epochs = bool(sequential_epochs)
        self._epoch_seed = int(epoch_seed)
        self._n_epoch = max(1, int(n_epoch))
        self._grpo_full_buffer = bool(grpo_full_buffer)

        prepared: List[Any] = []
        for ex in examples:
            e = dict(ex)
            e.pop("rollout_trajectory_id", None)
            prepared.append(e)
        examples = prepared

        # examples = [example for _ in range(repeat) for example in examples]
        collated = _data_loader._collate_fn(examples)

        def to_numpy_leaf(x: Any) -> Any:
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return np.asarray(x.detach().cpu())
            if isinstance(x, np.ndarray):
                return x
            if isinstance(x, jax.Array):
                return np.asarray(jax.device_get(x))
            return np.asarray(x)

        self._cpu_data: Any = jax.tree.map(to_numpy_leaf, collated)
        leaves = jax.tree.leaves(self._cpu_data)
        if not leaves:
            raise ValueError("examples must not be empty")
        self.num_examples = int(leaves[0].shape[0])
        if self.num_examples == 0:
            raise ValueError("examples must contain at least one sample")

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __len__(self) -> int:
        """Gradient steps per epoch (sequential) or per GRPO pooled-buffer schedule."""
        if self._grpo_full_buffer:
            return int(self._n_epoch)
        if not self._sequential_epochs:
            raise TypeError(
                "NeighborsDataLoader: __len__ is only defined when sequential_epochs=True "
                "or grpo_full_buffer=True"
            )
        return (self.num_examples + self.batch_size - 1) // self.batch_size

    def iter_global_microbatches(self):
        """Yield lists of microbatches: one list per shuffled full pass over **all** rows (``n_epoch`` passes).

        Each inner list partitions the pooled buffer (all rollouts) into non-overlapping microbatches
        of ``batch_size`` (last may be smaller). Caller (``train_model_on_fly``) averages gradients
        over the full list for one optimizer update per yield.
        """
        if not self._grpo_full_buffer:
            raise TypeError("iter_global_microbatches() requires grpo_full_buffer=True")
        for _ in range(self._n_epoch):
            rng = np.random.default_rng(self._epoch_seed)
            self._epoch_seed += 1
            perm = rng.permutation(self.num_examples)
            micros: list[
                Union[
                    tuple[_model.Observation, _model.Actions],
                    tuple[_model.Observation, _model.Actions, jax.Array],
                ]
            ] = []
            for start in range(0, self.num_examples, self.batch_size):
                chunk = perm[start : start + self.batch_size]
                micros.append(self._take_batch(chunk))
            yield micros

    def _take_batch(self, idx: np.ndarray) -> Union[
        tuple[_model.Observation, _model.Actions],
        tuple[_model.Observation, _model.Actions, jax.Array],
    ]:
        cpu_batch = jax.tree.map(
            lambda x: np.take(x, idx, axis=0) if x is not None else None,
            self._cpu_data,
        )
        batch = jax.tree.map(
            lambda x: jnp.asarray(x) if x is not None else None,
            cpu_batch,
        )
        obs = _model.Observation.from_dict(batch)
        act = batch["actions"]
        w = batch.get("distill_sample_weight")
        if w is not None:
            return (obs, act, w)
        return (obs, act)

    def __iter__(self) -> Iterator[
        Union[
            tuple[_model.Observation, _model.Actions],
            tuple[_model.Observation, _model.Actions, jax.Array],
        ]
    ]:
        """Yield minibatches: either infinite random (default) or one shuffled epoch."""
        if self._sequential_epochs:
            rng = np.random.default_rng(self._epoch_seed)
            self._epoch_seed += 1
            perm = rng.permutation(self.num_examples)
            for start in range(0, self.num_examples, self.batch_size):
                chunk = perm[start : start + self.batch_size]
                yield self._take_batch(chunk)
            return

        while True:
            # Sample a minibatch with replacement to keep a fixed shape.
            batch_idx = np.random.randint(0, self.num_examples, size=self.batch_size)
            cpu_batch = jax.tree.map(
                lambda x: np.take(x, batch_idx, axis=0) if x is not None else None,
                self._cpu_data,
            )
            batch = jax.tree.map(
                lambda x: jnp.asarray(x) if x is not None else None,
                cpu_batch,
            )
            obs = _model.Observation.from_dict(batch)
            act = batch["actions"]
            w = batch.get("distill_sample_weight")
            if w is not None:
                yield (obs, act, w)
            else:
                yield (obs, act)


def _interleave_pair_batch_leaves(opd_b: Any, aug_b: Any) -> Any:
    """Interleave batch dimension: (P,...) + (P,...) -> (2P,...) as o0,a0,o1,a1,..."""

    def interleave(o: jax.Array, a: jax.Array) -> jax.Array:
        stacked = jnp.stack([o, a], axis=1)
        p = int(o.shape[0])
        return stacked.reshape(2 * p, *tuple(o.shape[1:]))

    return jax.tree.map(interleave, opd_b, aug_b)


class PairedOnPolicySelfReplayDataLoader:
    """BC loader: each step samples P on-policy distillation (task2 teacher) rows and P paired
    same-observation rows with task1 prompt + reference-policy pseudo-actions; yields one batch
    of size 2P in PAIR_STREAM_INTERLEAVED order (even indices = OPD, odd = task1 pseudo)."""

    def __init__(
        self,
        opd_examples: List[Any],
        aug_examples: List[Any],
        pair_batch_size: int,
        data_config: _config.DataConfig,
    ):
        if len(opd_examples) != len(aug_examples) or len(opd_examples) == 0:
            raise ValueError("opd_examples and aug_examples must have the same non-empty length")
        if pair_batch_size <= 0:
            raise ValueError(f"pair_batch_size must be > 0, got {pair_batch_size}")
        self._data_config = data_config
        self.pair_batch_size = pair_batch_size
        self.n = len(opd_examples)

        def to_jax(x: Any) -> jax.Array:
            if isinstance(x, torch.Tensor):
                return jnp.asarray(x)
            if isinstance(x, np.ndarray):
                return jnp.asarray(x)
            if isinstance(x, jax.Array):
                return x
            return jnp.asarray(x)

        self.opd_data = jax.tree.map(to_jax, _data_loader._collate_fn(opd_examples))
        self.aug_data = jax.tree.map(to_jax, _data_loader._collate_fn(aug_examples))
        leaves = jax.tree.leaves(self.opd_data)
        if not leaves:
            raise ValueError("opd_examples collate produced empty tree")
        if int(leaves[0].shape[0]) != self.n:
            raise ValueError("internal: collated batch size mismatch")

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self) -> Iterator[
        Union[
            tuple[_model.Observation, _model.Actions],
            tuple[_model.Observation, _model.Actions, jax.Array],
        ]
    ]:
        while True:
            idx = np.random.randint(0, self.n, size=self.pair_batch_size)
            opd_b = jax.tree.map(lambda x: jnp.take(x, idx, axis=0), self.opd_data)
            aug_b = jax.tree.map(lambda x: jnp.take(x, idx, axis=0), self.aug_data)
            merged = _interleave_pair_batch_leaves(opd_b, aug_b)
            obs = _model.Observation.from_dict(merged)
            act = merged["actions"]
            w = merged.get("distill_sample_weight")
            if w is not None:
                yield (obs, act, w)
            else:
                yield (obs, act)


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

    # Create a fresh copy of the original model for training.
    # Must deep-copy params so model_phase2 has independent buffers; otherwise
    # init_train_state (donate_argnums=5) invalidates shared buffers and
    # model_phase1 (used for augment_inference_fn) would produce garbage.
    def copy_param(x):
        if isinstance(x, jax.Array):
            jax.block_until_ready(x)
            return jnp.array(np.asarray(x))  # Force copy through CPU
        if hasattr(x, "value") and hasattr(x, "replace"):
            # nnx.Param / nnx.Variable
            jax.block_until_ready(x.value)
            new_val = jnp.array(np.asarray(x.value))
            return x.replace(value=new_val)
        return x

    original_params_copy = jax.tree.map(copy_param, original_model_params)
    jax.block_until_ready(original_params_copy)
    model_copy = nnx.merge(original_model_graphdef, original_params_copy)
    model_copy.eval()
    # NOTE: No need to block here - nnx.merge() doesn't create new arrays, just combines already-materialized params
    return model_copy


def merge_model_parameters(
    trained_model: _model.BaseModel,
    original_model: _model.BaseModel,
    merging_eps: float,
) -> _model.BaseModel:
    """Blend trained/original params while reusing trained_model storage."""
    if not (0.0 <= merging_eps <= 1.0):
        raise ValueError(f"merging_eps must be in [0, 1], got {merging_eps}")

    @functools.partial(jax.jit, donate_argnums=(0,))
    def _blend_params_donate(
        trained_params: Any,
        original_params: Any,
        eps: float,
    ) -> Any:
        return jax.tree.map(
            lambda t, o: eps * t + (1.0 - eps) * o,
            trained_params,
            original_params,
        )

    trained_state = nnx.state(trained_model)
    original_state = nnx.state(original_model)

    trained_params = jax.tree.map(
        lambda leaf: leaf.value if hasattr(leaf, "value") else leaf,
        trained_state,
    )
    original_params = jax.tree.map(
        lambda leaf: leaf.value if hasattr(leaf, "value") else leaf,
        original_state,
    )

    blended_params = _blend_params_donate(trained_params, original_params, merging_eps)
    blended_state = jax.tree.map(
        lambda leaf, value: (
            # Important: for non-Variable leaves, use `value` (the blended tensor),
            # not `leaf`, because `leaf` may reference donated/deleted buffers.
            leaf.replace(value=value) if hasattr(leaf, "value") else value
        ),
        trained_state,
        blended_params,
    )
    nnx.update(trained_model, blended_state)
    trained_model.eval()
    return trained_model


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
    ttt_k: int = 6,
    num_neighbors_fetch: int = 32,
    ttt_use_modalities: Optional[List[str]] = None,
    plot_observations: bool = False,
    disable_adaptation: bool = False,
    repeat_batch: int = 1,
    meta_update: str = "reset",
    no_reset: bool = False,
    use_test_task: bool = False,
    random_neighbors: bool = False,
    cfg_weight: float = 1.0,
    num_samples: int = 1,
    debug_metrics: bool = False,
    merging_eps: float | None = None,
    teacher_policy: _policy.Policy | None = None,
    show_progress_bar: bool = True,
    distillation_examples_out: list[dict[str, Any]] | None = None,
    student_action_merge: float = 0.0,
    temporal_decay: float = 1.0,
    teacher_group_size: int = 1,
    write_auxiliary_rollout_pdfs: bool = True,
    after_each_rollout_episode: Callable[[int, int, int], None] | None = None,
    grpo_like: bool = False,
    grpo_trust_eps: float | None = None,
    grpo_weight: str | None = None,
    grpo_weight_eps: float = 1e-8,
    distill_collect_every: int = 1,
    distill_trajectory_id_offset: int = 0,
    num_envs: int = 1,
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
        num_neighbors_fetch=num_neighbors_fetch,
        ttt_use_modalities=ttt_use_modalities,
        plot_observations=plot_observations,
        repeat_batch=repeat_batch,
        meta_update=meta_update,
        no_reset=no_reset,
        use_test_task=use_test_task,
        random_neighbors=random_neighbors,
        cfg_weight=cfg_weight,
        num_samples=num_samples,
        debug_metrics=debug_metrics,
        merging_eps=merging_eps,
        adapt_kwargs={},
        disable_adaptation=disable_adaptation,
        teacher_policy=teacher_policy,
        show_progress_bar=show_progress_bar,
        distillation_examples_out=distillation_examples_out,
        student_action_merge=student_action_merge,
        temporal_decay=temporal_decay,
        teacher_group_size=teacher_group_size,
        write_auxiliary_rollout_pdfs=write_auxiliary_rollout_pdfs,
        after_each_rollout_episode=after_each_rollout_episode,
        grpo_like=grpo_like,
        grpo_trust_eps=grpo_trust_eps,
        grpo_weight=grpo_weight,
        grpo_weight_eps=grpo_weight_eps,
        distill_collect_every=distill_collect_every,
        distill_trajectory_id_offset=distill_trajectory_id_offset,
        num_envs=num_envs,
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
        meta_update="reset" if reset_policy else "continual_ttt",
        adapt_kwargs=dict(noise_std=noise_sigma),
    )


def _infer_action_chunk_samples(
    policy_obj: _policy.Policy,
    obs_dict: dict[str, Any],
    noise_arr: np.ndarray | jnp.ndarray | None,
) -> _model.Actions:
    """Infer action chunk(s). ``noise`` (H, D) or (1, H, D): one chunk (H, D).

    ``noise`` (G, H, D) with G>1: G independent samples at the same obs; return (G, H, D).
    """
    if noise_arr is None:
        return policy_obj.infer(obs_dict, noise=None)["actions"]
    n = jnp.asarray(noise_arr)
    if n.ndim == 2:
        return policy_obj.infer(obs_dict, noise=n)["actions"]
    if n.ndim != 3:
        raise ValueError(f"noise must be (H,D), (1,H,D), or (G,H,D); got shape {n.shape}")
    g = int(n.shape[0])
    if g == 1:
        return policy_obj.infer(obs_dict, noise=n[0])["actions"]
    # One batched forward for G samples at the same observation (JAX).
    if not getattr(policy_obj, "_is_pytorch_model", False) and hasattr(
        policy_obj, "infer_batch"
    ):
        acts = policy_obj.infer_batch([obs_dict] * g, noise=n)["actions"]
        return jnp.asarray(acts)
    action_chunks: list[jax.Array] = []
    for gi in range(g):
        action_i = policy_obj.infer(obs_dict, noise=n[gi])["actions"]
        action_chunks.append(
            action_i if isinstance(action_i, jax.Array) else jnp.asarray(action_i)
        )
    return jnp.stack(action_chunks, axis=0)


def _infer_grouped_batch_multi_env(
    policy_obj: _policy.Policy,
    obs_list: list[dict[str, Any]],
    noises_per_env: list[jnp.ndarray],
) -> jnp.ndarray:
    """Batched forward for B envs, each with G noise rows (same obs repeated G times).

    Returns actions of shape ``(B, G, H, D)``.
    """
    if not obs_list:
        raise ValueError("empty obs_list")
    b = len(obs_list)
    if len(noises_per_env) != b:
        raise ValueError("noises_per_env length must match obs_list")
    obs_exp: list[dict[str, Any]] = []
    noise_blocks: list[jnp.ndarray] = []
    g_sizes: list[int] = []
    for i in range(b):
        n = jnp.asarray(noises_per_env[i])
        if n.ndim == 2:
            n = n[None, ...]
        g = int(n.shape[0])
        g_sizes.append(g)
        for _ in range(g):
            obs_exp.append(obs_list[i])
        noise_blocks.append(n)
    if len(set(g_sizes)) != 1:
        raise ValueError(
            f"infer_grouped_batch_multi_env requires the same G for all envs; got {g_sizes}"
        )
    g0 = g_sizes[0]
    noise_flat = jnp.concatenate(noise_blocks, axis=0)
    if not getattr(policy_obj, "_is_pytorch_model", False) and hasattr(
        policy_obj, "infer_batch"
    ):
        acts = policy_obj.infer_batch(obs_exp, noise=noise_flat)["actions"]
        acts = jnp.asarray(acts)
        return acts.reshape(b, g0, acts.shape[1], acts.shape[2])
    # Fallback: sequential infer per row
    out_rows: list[jnp.ndarray] = []
    row = 0
    for i in range(b):
        for _gi in range(g0):
            out_rows.append(
                jnp.asarray(
                    policy_obj.infer(obs_exp[row], noise=noise_flat[row])["actions"]
                )
            )
            row += 1
    stacked = jnp.stack(out_rows, axis=0)
    return stacked.reshape(b, g0, stacked.shape[1], stacked.shape[2])


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
    filename_prefix: str = "losses",
    y_label: str = "Loss",
) -> None:
    if not episode_losses or ttt_num_steps <= 1:
        return
    if not any(len(step_losses) > 0 for step_losses in episode_losses):
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
        if step_losses:
            ax.plot(range(len(step_losses)), step_losses, "b-o", markersize=2)
        ax.set_title(f"TTT update {i+1} ({len(step_losses)} steps)")
        ax.set_xlabel("Gradient step")
        ax.set_ylabel(y_label)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle(f"Episode {episode_idx+1} – {'success' if done else 'failure'}")
    plt.tight_layout()
    os.makedirs(pathlib.Path(video_out_path).parent / "plots_losses", exist_ok=True)
    losses_plot_path = pathlib.Path(video_out_path).parent / "plots_losses" / f"{filename_prefix}_ep_{episode_idx+1}.pdf"
    plt.savefig(losses_plot_path, bbox_inches="tight")
    plt.close(fig)


def _neighbor_prompt_text(example: Any) -> str:
    """Best-effort prompt extraction from a neighbor sample."""
    if isinstance(example, dict):
        prompt = example.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()

        tok = example.get("tokenized_prompt")
        tok_mask = example.get("tokenized_prompt_mask")
        if tok is not None:
            try:
                tok_arr = np.asarray(tok)
                tok_mask_arr = np.asarray(tok_mask) if tok_mask is not None else None
                if tok_arr.ndim > 1:
                    tok_arr = tok_arr[0]
                if tok_mask_arr is not None and tok_mask_arr.ndim > 1:
                    tok_mask_arr = tok_mask_arr[0]
                return decode_tokenized_prompt(
                    tokenized_prompt=tok_arr,
                    tokenized_prompt_mask=tok_mask_arr,
                    max_token_len=200,
                    fallback_text="(prompt unavailable)",
                )
            except Exception:
                pass
    return "(prompt unavailable)"


def _neighbors_to_preview_data(
    examples: list[Any],
    *,
    top_n: int = 4,
) -> list[dict[str, Any]]:
    """Build top-N preview entries: composite image + decoded prompt."""
    if not examples:
        return []

    top_examples = examples[:top_n]
    batch = _data_loader._collate_fn(top_examples)

    def to_jax(x):
        if isinstance(x, torch.Tensor):
            return jnp.asarray(x)
        if isinstance(x, np.ndarray):
            return jnp.asarray(x)
        if isinstance(x, jax.Array):
            return x
        return jnp.asarray(x)

    batch = jax.tree.map(to_jax, batch)
    obs = _model.Observation.from_dict(batch)
    base_imgs = obs.images.get("base_0_rgb")
    wrist_imgs = obs.images.get("left_wrist_0_rgb")
    if base_imgs is None or wrist_imgs is None:
        return []

    preview_data: list[dict[str, Any]] = []
    for i in range(min(top_n, int(base_imgs.shape[0]))):
        base = np.asarray(base_imgs[i], dtype=np.float32)
        wrist = np.asarray(wrist_imgs[i], dtype=np.float32)
        base = np.clip((base + 1.0) / 2.0, 0.0, 1.0)
        wrist = np.clip((wrist + 1.0) / 2.0, 0.0, 1.0)
        preview_data.append(
            {
                "composite": np.concatenate([base, wrist], axis=1),
                "prompt": _neighbor_prompt_text(top_examples[i]),
            }
        )
    return preview_data


def _save_episode_neighbors_plot(
    *,
    neighbor_previews: list[dict[str, Any]],
    video_out_path: str,
    episode_idx: int,
    done: bool,
) -> None:
    """Save per-update neighbor preview grid (top-4 each update)."""
    if not neighbor_previews:
        return

    n_updates = len(neighbor_previews)
    n_cols = 4
    fig, axes = plt.subplots(
        n_updates, n_cols, figsize=(3.2 * n_cols, 2.4 * n_updates), squeeze=False
    )

    for row, preview in enumerate(neighbor_previews):
        items: list[dict[str, Any]] = preview.get("items", [])
        top_similarities: list[float] = preview.get("top_similarities", [])
        max_sim = float(preview.get("max_similarity", float("nan")))
        min_sim = float(preview.get("min_similarity", float("nan")))

        for col in range(n_cols):
            ax = axes[row, col]
            if col < len(items):
                item = items[col]
                img = item.get("composite")
                prompt = str(item.get("prompt", "(prompt unavailable)"))
                sim = top_similarities[col] if col < len(top_similarities) else float("nan")
                ax.imshow(img)
                title_prompt = (prompt[:60] + "...") if len(prompt) > 60 else prompt
                ax.set_title(f"NN {col+1} | sim={sim:.4f}\n{title_prompt}", fontsize=7)
            ax.axis("off")

        axes[row, 0].text(
            0.0,
            1.06,
            f"TTT {row+1} | sim range: [{min_sim:.4f}, {max_sim:.4f}]",
            transform=axes[row, 0].transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    status = "success" if done else "failure"
    plt.suptitle(f"Episode {episode_idx+1} – {status} – Neighbor previews", fontsize=11)
    plt.tight_layout()
    out_dir = pathlib.Path(video_out_path).parent / "plots_neighbors"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"neighbors_ep_{episode_idx+1}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
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


def compute_action_distances(action_chunk: _model.Actions, ttt_action_chunk: _model.Actions) -> float:
    # Compute distance between action chunk and ttt action chunk
    def preprocess_action(action: _model.Actions) -> jax.Array:
        action = action if isinstance(action, jax.Array) else jnp.array(action)
        if action.ndim == 2:
            action = action[None, ...]
        return action

    action_chunk = preprocess_action(action_chunk)
    ttt_action_chunk = preprocess_action(ttt_action_chunk)
    action_distance_per_sample = jnp.linalg.norm(action_chunk - ttt_action_chunk, axis=(1, 2))
    action_distance = float(jax.device_get(jnp.mean(action_distance_per_sample)))

    return action_distance


def _pad_action_chunk_for_model(
    actions_np: np.ndarray,
    *,
    action_horizon: int,
    action_dim: int,
) -> np.ndarray:
    """Match LeRobot/OpenPI training: (H, D) with H=action_horizon, D=action_dim (zero-pad)."""
    a = np.asarray(actions_np, dtype=np.float32)
    if a.ndim == 3 and int(a.shape[0]) == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(
            f"Expected teacher actions with shape (H, D) or (1, H, D), got {a.shape}"
        )
    h, d = int(a.shape[0]), int(a.shape[1])
    if h > action_horizon:
        a = a[:action_horizon]
    elif h < action_horizon:
        a = np.pad(a, ((0, action_horizon - h), (0, 0)), constant_values=0.0)
        d = int(a.shape[1])
    if d > action_dim:
        a = a[:, :action_dim]
    else:
        a = transforms.pad_to_dim(a, action_dim, axis=-1)
    return a


def _compute_grpo_advantages(
    student_stacked: jax.Array,
    teacher_chunk: jax.Array,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """GRPO-style advantages from negative L2 distance to teacher chunk.

    R_i = -||a_i - a_teach||_2 (flattened). A_hat_i = (R_i - mean(R)) / std(R), or zeros if std ~ 0.
    Returns shape (G,) float32 on CPU.
    """
    st = jnp.asarray(student_stacked)
    tt = jnp.asarray(teacher_chunk)
    if st.ndim == 2:
        st = st[None, ...]
    if tt.ndim == 3:
        tt = tt[0]
    g = int(st.shape[0])
    if g < 1:
        return np.zeros((0,), dtype=np.float32)
    diff = st - tt[None, ...]
    dists = jnp.linalg.norm(diff.reshape(g, -1), axis=1)
    r = -dists
    mean_r = jnp.mean(r)
    std_r = jnp.std(r)
    a_hat = jnp.where(
        std_r > eps,
        (r - mean_r) / std_r,
        jnp.zeros_like(r),
    )
    return np.asarray(jax.device_get(a_hat), dtype=np.float32)


def _compute_grpo_dists_np(
    student_stacked: jax.Array,
    teacher_chunk: jax.Array,
) -> np.ndarray:
    """Per-sample L2 norms ||a_i - a_teach|| (flattened chunk), shape (G,) float32."""
    st = jnp.asarray(student_stacked)
    tt = jnp.asarray(teacher_chunk)
    if st.ndim == 2:
        st = st[None, ...]
    if tt.ndim == 3:
        tt = tt[0]
    g = int(st.shape[0])
    if g < 1:
        return np.zeros((0,), dtype=np.float32)
    diff = st - tt[None, ...]
    dists = jnp.linalg.norm(diff.reshape(g, -1), axis=1)
    return np.asarray(jax.device_get(dists), dtype=np.float32)


def _compute_grpo_mean_std_weight(
    dists: np.ndarray | Sequence[float],
    *,
    eps: float = 1e-8,
) -> float:
    """w_t = mean(||d||) / (std(||d||) + eps); amplifies when mean error is high vs spread.

    Returns 1.0 when fewer than two distances (undefined spread).
    """
    d = np.asarray(dists, dtype=np.float64).ravel()
    if d.size < 2:
        return 1.0
    m = float(np.mean(d))
    s = float(np.std(d))
    return float(m / (s + eps))


def _grpo_trust_clamp_target_action(
    a_stud: jax.Array,
    a_teach: jax.Array,
    trust_eps: float | None,
) -> jax.Array:
    """Trust-region BC target: a_target = a_stud + clamp(a_teach - a_stud, -ε, ε) (element-wise).

    When ``trust_eps`` is None, returns ``a_stud`` unchanged (raw student chunk for GRPO).
    """
    if trust_eps is None:
        return a_stud
    a = jnp.asarray(a_stud)
    t = jnp.asarray(a_teach)
    if t.ndim == 3:
        t = t[0]
    delta = t - a
    te = float(trust_eps)
    return a + jnp.clip(delta, -te, te)


def observation_actions_to_bc_example(
    obs: _model.Observation,
    actions: jax.Array | np.ndarray,
    *,
    model_action_horizon: int | None = None,
    model_action_dim: int | None = None,
    alignment_ratio: float | None = None,
    replan_env_step: int | None = None,
    temporal_decay: float = 1.0,
    teacher_chunk_var_mean: float | None = None,
    grpo_advantage: float | None = None,
    grpo_group_weight: float | None = None,
    rollout_trajectory_id: int | None = None,
) -> dict[str, Any]:
    """Build one OpenPI-style sample dict for ``_data_loader._collate_fn`` (no batch dimension).

    Observations must come from ``make_observation_from_simulator(student_policy, ...)`` so
    transforms match training. ``actions`` are distillation targets (full action chunk), typically
    teacher-only or a ``(1-α)*teacher + α*student`` blend when ``student_action_merge`` is used.

    Pi05 Libero uses ``action_dim=32`` with zero-padded proprio/actions (see ``PadStatesAndActions``).
    Raw policy chunks are often ``(horizon, 7)``; pass ``model_action_*`` so BC batches match ``compute_loss``.

    Optional ``alignment_ratio`` / ``replan_env_step`` / ``teacher_chunk_var_mean`` are metadata for
    distillation filtering and plots; strip these keys before ``NeighborsDataLoader`` / ``Observation.from_dict``.
    ``teacher_chunk_var_mean`` is mean_{h,d} Var_{teacher samples}(chunk[h,d]) when ``teacher_group_size``>1, else 0.

    ``temporal_decay``: per-sample BC weight ``temporal_decay ** replan_env_step`` (env step at replan).
    Default ``1.0`` gives uniform weights. Stored as ``distill_sample_weight`` (kept for the BC loader).

    If ``grpo_advantage`` is set (GRPO-like training), ``distill_sample_weight`` is
    ``grpo_advantage * temporal_decay ** replan_env_step`` (signed advantage times temporal factor).
    Optional ``grpo_group_weight`` (default 1) multiplies that product when GRPO is on, e.g.
    mean/std weighting over the group's L2 distances to the teacher.
    """

    def _strip_leading_batch(x: Any) -> Any:
        if x is None:
            return None
        arr = np.asarray(jax.device_get(x))
        if arr.ndim >= 1 and int(arr.shape[0]) == 1:
            return np.squeeze(arr, axis=0)
        return arr

    act = jnp.asarray(actions)
    if act.ndim == 3 and int(act.shape[0]) == 1:
        act = act[0]
    actions_np = np.asarray(jax.device_get(act), dtype=np.float32)

    tree = obs.to_dict()
    out: dict[str, Any] = jax.tree.map(
        _strip_leading_batch, tree, is_leaf=lambda x: x is None
    )
    if (
        model_action_horizon is not None
        and model_action_dim is not None
    ):
        actions_np = _pad_action_chunk_for_model(
            actions_np,
            action_horizon=model_action_horizon,
            action_dim=model_action_dim,
        )
        if out.get("state") is not None:
            st = np.asarray(out["state"], dtype=np.float32)
            if st.shape[-1] > model_action_dim:
                st = st[..., :model_action_dim]
            else:
                st = transforms.pad_to_dim(st, model_action_dim, axis=-1)
            out["state"] = st
    out["actions"] = actions_np
    if alignment_ratio is not None:
        out["alignment_ratio"] = np.float32(alignment_ratio)
    gw = 1.0 if grpo_group_weight is None else float(grpo_group_weight)
    if replan_env_step is not None:
        out["replan_env_step"] = np.int32(replan_env_step)
        td = float(temporal_decay)
        step_i = int(replan_env_step)
        if grpo_advantage is not None:
            out["distill_sample_weight"] = np.float32(
                float(grpo_advantage) * gw * (td**step_i)
            )
        else:
            out["distill_sample_weight"] = np.float32(td**step_i)
    elif grpo_advantage is not None:
        out["distill_sample_weight"] = np.float32(float(grpo_advantage) * gw)
    if teacher_chunk_var_mean is not None:
        out["teacher_chunk_var_mean"] = np.float32(float(teacher_chunk_var_mean))
    if rollout_trajectory_id is not None:
        out["rollout_trajectory_id"] = np.int32(int(rollout_trajectory_id))
    return out


def _plot_teacher_student_action_l2_pdf(
    episodes_l2: list[list[tuple[int, float]]],
    output_path: str | pathlib.Path,
    *,
    task_id: int,
) -> pathlib.Path | None:
    """One subplot per episode: env step vs L2 between teacher and student action chunks."""
    if not episodes_l2 or all(len(ep) == 0 for ep in episodes_l2):
        return None
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(episodes_l2)
    ncols = min(3, max(1, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False
    )
    for i, series in enumerate(episodes_l2):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if series:
            xs = [p[0] for p in series]
            ys = [p[1] for p in series]
            ax.plot(xs, ys, "b.-", linewidth=1.2, markersize=4)
        ax.set_xlabel("Environment step")
        ax.set_ylabel("L2 distance (teacher vs student)")
        ax.set_title(f"Episode {i + 1}")
        ax.grid(True, alpha=0.3)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle(
        f"Teacher vs student action chunk distance (task {task_id})",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(path, format="pdf", dpi=150)
    plt.close(fig)
    return path


def _plot_group_action_sampling_pdf(
    trace: list[dict[str, Any]],
    output_path: str | pathlib.Path,
    *,
    episode_idx: int,
    group_size: int,
) -> pathlib.Path | None:
    """One PDF per rollout episode: chunk variance, teacher L2 distances, GRPO-style normalized distances."""
    if not trace:
        return None
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [int(x["env_step"]) for x in trace]
    vars_ = [float(x["chunk_var_mean"]) for x in trace]
    all_x: list[int] = []
    all_d: list[float] = []
    all_n: list[float] = []
    for x in trace:
        t = int(x["env_step"])
        for d, nz in zip(x["dists"], x["norm_dists"]):
            all_x.append(t)
            all_d.append(float(d))
            all_n.append(float(nz))
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.5), sharex=True)
    fig.suptitle(
        f"Group sampling (G={group_size}) — episode {episode_idx}",
        fontsize=12,
    )
    axes[0].plot(steps, vars_, "b.-", linewidth=1.0, markersize=5)
    axes[0].set_ylabel("Mean Var across chunk\n(var over samples)")
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(all_x, all_d, s=14, alpha=0.65, c="C0", edgecolors="none")
    axes[1].set_ylabel(r"$\|a_g - a^{\mathrm{teacher}}\|_2$ (flat)")
    axes[1].grid(True, alpha=0.3)
    axes[2].scatter(all_x, all_n, s=14, alpha=0.65, c="C2", edgecolors="none")
    axes[2].set_ylabel(r"Normalized: $(d - \bar d) / (\sigma_d+\epsilon)$")
    axes[2].set_xlabel("Environment step (replan)")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, format="pdf", dpi=150)
    plt.close(fig)
    return path


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
    num_neighbors_fetch: int,
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
    merging_eps: float | None,
    adapt_kwargs: dict[str, Any],
    meta_update: str,
    no_reset: bool,
    cfg_weight: float,
    ttt_count: int,
    jax_key: jax.Array,
    pbar: Any,
    teacher_policy: _policy.Policy | None = None,
    distillation_examples_out: list[dict[str, Any]] | None = None,
    student_action_merge: float = 0.0,
    temporal_decay: float = 1.0,
    teacher_group_size: int = 1,
    grpo_like: bool = False,
    grpo_trust_eps: float | None = None,
    grpo_weight: str | None = None,
    grpo_weight_eps: float = 1e-8,
    distill_collect_every: int = 1,
    rollout_trajectory_id: int = 0,
) -> tuple[
    bool,
    int,
    list[np.ndarray],
    list[tuple[int, float]],
    list[tuple[int, float]],
    list[list[float]],
    list[list[float]],
    list[dict[str, Any]],
    list[tuple[int, float]],
    int,
    _policy.Policy,
    jax.Array,
    list[tuple[int, float]],
    list[dict[str, Any]],
    list[tuple[int, float]],
    list[dict[str, Any]],
]:
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
    episode_test_losses: list[list[float]] = []
    neighbor_previews: list[dict[str, Any]] = []
    teacher_l2_by_env_step: list[tuple[int, float]] = []
    group_sampling_trace: list[dict[str, Any]] = []
    teacher_alignment_ratio_by_step: list[tuple[int, float]] = []
    teacher_group_sampling_trace: list[dict[str, Any]] = []

    if num_samples > 1 and not disable_adaptation:
        raise ValueError(
            "num_samples > 1 (group action sampling) requires disable_adaptation=True "
            "(e.g. standard distillation rollout)."
        )

    if not no_reset:
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

        # Images are flipped horizontally and vertically and resized
        img, wrist_img = _preprocess_images(obs, resize_size=resize_size)
        replay_images.append(_draw_step_on_frame(img, t))

        replan_now = not action_plan

        if not action_plan:

            # Packing the observation dictionary (no image processing)
            curr_obs_dict = _build_curr_obs_dict(obs, img, wrist_img, task_description)
            policy._rng, rng = jax.random.split(policy._rng)

            noise = jax.random.normal(
                rng, (num_samples, original_model.action_horizon, original_model.action_dim)
            )

            # Sample action chunk(s) from policy (G>1 → stacked (G,H,D); env uses first sample)
            student_stacked = _infer_action_chunk_samples(policy, curr_obs_dict, noise)
            if student_stacked.ndim == 3:
                action_chunk = student_stacked[0]
            else:
                action_chunk = student_stacked

            # Compute alignment ratio between action chunk and the action the model would have sampled with empty prompt
            # action_chunk_cfg is the action taken with classifier free guidance ()
            alignment_ratio = compute_alignment_ratio(
                policy,
                action_chunk,
                curr_obs_dict,
                noise=noise,
                cfg_weight=cfg_weight,
            )
            # Enable in case of debugging
            # pbar.write(f"[TTT] Alignment ratio: {alignment_ratio:.4f}")
            alignment_ratio_by_step.append((t, alignment_ratio))


            assert len(action_chunk) >= replan_steps, (
                f"Policy only predicts {len(action_chunk)} steps, need {replan_steps}"
            )

            if not disable_adaptation and (t - num_steps_wait) % ttt_frequency == 0 and t > max_ttt_step:

                observation = make_observation_from_simulator(policy, curr_obs_dict)
                _unused_examples, distances = nn_lookup(
                    observation=observation,
                    nn_fetcher=nn_fetcher,
                    dataset=dataset,
                    use_modalities=ttt_use_modalities,
                    k=num_neighbors_fetch,
                    repeat_batch=repeat_batch,
                    plot_observations=plot_observations,
                    use_test_task=use_test_task,
                    pbar=pbar,
                )
                del _unused_examples, distances

            if not disable_adaptation and (t - num_steps_wait) % ttt_frequency == 0 and t <= max_ttt_step:
                ttt_count += 1

                observation = make_observation_from_simulator(policy, curr_obs_dict)
                ttt_training_data, distances = nn_lookup(
                    observation=observation,
                    nn_fetcher=nn_fetcher,
                    dataset=dataset,
                    use_modalities=ttt_use_modalities,
                    k=num_neighbors_fetch,
                    repeat_batch=repeat_batch,
                    plot_observations=plot_observations,
                    use_test_task=use_test_task,
                    pbar=pbar,
                    random_neighbors=random_neighbors,
                )

                adaptation_kwargs = dict(adapt_kwargs)
                adaptation_kwargs.setdefault("batch_size", ttt_k)
                distances_np = np.asarray(distances, dtype=np.float32)
                top_similarities = [float(v) for v in distances_np[:4].tolist()] if distances_np.size > 0 else []
                if plot_observations:
                    neighbor_previews.append(
                        {
                            "items": _neighbors_to_preview_data(ttt_training_data, top_n=4),
                            "top_similarities": top_similarities,
                            "max_similarity": float(np.max(distances_np)) if distances_np.size > 0 else float("nan"),
                            "min_similarity": float(np.min(distances_np)) if distances_np.size > 0 else float("nan"),
                        }
                    )

                trained_model, losses = adaptation_fn(
                    policy=policy,
                    train_config=train_config,
                    ttt_data_config=ttt_data_config,
                    ttt_training_data=ttt_training_data,
                    learning_rate=learning_rate,
                    num_steps=ttt_num_steps,
                    warmup_steps=0,
                    weight_decay=0.0,
                    log_interval=max(1, ttt_num_steps // 2),
                    seed=episode_seed + ttt_count,
                    pbar=pbar,
                    num_samples=num_samples,
                    debug_metrics=debug_metrics,
                    **adaptation_kwargs,
                )

                ttt_policy = new_policy_like(policy, trained_model)
                ttt_action_chunk = _infer_action_chunk_samples(ttt_policy, curr_obs_dict, noise)

                # Compute action distances
                action_distance = compute_action_distances(action_chunk, ttt_action_chunk)

                distances_actions.append((t, action_distance))
                similarities.append((t, distances[0]))
                episode_losses.append(list(losses))
                test_losses = getattr(adaptation_fn, "last_test_losses", [])
                episode_test_losses.append(list(test_losses) if isinstance(test_losses, list) else [])
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
                action_chunk = ttt_action_chunk


                ### Meta update
                if meta_update == "reset":
                    del trained_model
                elif meta_update == "tt_reptile":
                    assert merging_eps is not None
                    trained_model = merge_model_parameters(
                        trained_model=trained_model,
                        original_model=original_model,
                        merging_eps=merging_eps,
                    )
                    policy = new_policy_like(policy, trained_model)
                elif meta_update == "continual_ttt":
                    policy = new_policy_like(policy, trained_model)
                else:  # pragma: no cover
                    raise ValueError(f"Unknown meta_update mode: {meta_update}")

                del losses, ttt_training_data, ttt_policy
                print_memory_checkpoint(
                    f"[TTT {ttt_count}] At the end of the TTT update deleting objects", 1300
                )

            if teacher_policy is not None:
                tgs = max(1, int(teacher_group_size))
                # teacher_group_size==1: same noise as student's first draw (legacy single-sample behavior).
                if tgs == 1:
                    teacher_noise = noise[0:1]
                else:
                    policy._rng, rng_teach = jax.random.split(policy._rng)
                    teacher_noise = jax.random.normal(
                        rng_teach,
                        (tgs, original_model.action_horizon, original_model.action_dim),
                    )
                teacher_stacked = _infer_action_chunk_samples(
                    teacher_policy, curr_obs_dict, teacher_noise
                )
                if teacher_stacked.ndim == 3:
                    teacher_chunk = teacher_stacked[0]
                else:
                    teacher_chunk = teacher_stacked

                tar = compute_alignment_ratio(
                    teacher_policy,
                    teacher_chunk,
                    curr_obs_dict,
                    noise=teacher_noise,
                )
                teacher_alignment_ratio_by_step.append((t, float(tar)))

                l2_ts = compute_action_distances(teacher_chunk, action_chunk)
                teacher_l2_by_env_step.append((t, l2_ts))
                if num_samples > 1 and student_stacked.ndim == 3:
                    st = jnp.asarray(student_stacked)
                    tt = jnp.asarray(teacher_chunk)
                    if tt.ndim == 3:
                        tt = tt[0]
                    diff = st - tt[None, ...]
                    dists_vec = jnp.linalg.norm(diff.reshape(st.shape[0], -1), axis=1)
                    dists_list = [float(v) for v in jax.device_get(dists_vec)]
                    chunk_var_mean = float(jnp.mean(jnp.var(st, axis=0)))
                    dm = float(np.mean(dists_list))
                    ds = float(np.std(dists_list)) + 1e-8
                    norm_list = [float((d - dm) / ds) for d in dists_list]
                    grpo_ms_w = (
                        _compute_grpo_mean_std_weight(dists_list, eps=grpo_weight_eps)
                        if len(dists_list) >= 2
                        else None
                    )
                    adv_arr = _compute_grpo_advantages(st, teacher_chunk)
                    grpo_adv_list = [float(v) for v in adv_arr]
                    group_sampling_trace.append(
                        {
                            "env_step": int(t),
                            "chunk_var_mean": chunk_var_mean,
                            "dists": dists_list,
                            "norm_dists": norm_list,
                            "grpo_mean_std_weight": grpo_ms_w,
                            "grpo_advantages": grpo_adv_list,
                        }
                    )
                else:
                    group_sampling_trace.append(
                        {
                            "env_step": int(t),
                            "chunk_var_mean": 0.0,
                            "dists": [],
                            "norm_dists": [],
                            "grpo_mean_std_weight": None,
                            "grpo_advantages": [],
                        }
                    )
                teacher_var_row = 0.0
                if tgs > 1 and teacher_stacked.ndim == 3:
                    tt2 = jnp.asarray(teacher_stacked)
                    teacher_var_row = float(jnp.mean(jnp.var(tt2, axis=0)))
                    teacher_group_sampling_trace.append(
                        {
                            "env_step": int(t),
                            "chunk_var_mean": teacher_var_row,
                        }
                    )
                else:
                    teacher_group_sampling_trace.append(
                        {"env_step": int(t), "chunk_var_mean": 0.0},
                    )

            action_plan.extend(action_chunk[:replan_steps])

        # On-policy distillation BC: one row per env step (current obs), while the env still
        # executes actions from the 5-step open-loop chunk. Reuse forwards on replan boundaries.
        if (
            teacher_policy is not None
            and distillation_examples_out is not None
            and t >= num_steps_wait
            and (t - num_steps_wait) % int(distill_collect_every) == 0
        ):
            if grpo_like:
                if num_samples < 2:
                    raise ValueError(
                        "grpo_like distillation requires num_samples (group_size) >= 2 "
                        "to form per-group advantages"
                    )
            alpha = float(student_action_merge)
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(
                    f"student_action_merge must be in [0, 1], got {alpha}"
                )
            if grpo_like:
                if replan_now:
                    st_d = jnp.asarray(student_stacked)
                    adv_d = _compute_grpo_advantages(st_d, teacher_chunk)
                    dists_np = _compute_grpo_dists_np(st_d, teacher_chunk)
                    w_grpo = (
                        _compute_grpo_mean_std_weight(dists_np, eps=grpo_weight_eps)
                        if grpo_weight == "mean_std"
                        else None
                    )
                    obs_m_d = make_observation_from_simulator(policy, curr_obs_dict)
                    for i_gr in range(int(st_d.shape[0])):
                        tgt_gr = _grpo_trust_clamp_target_action(
                            st_d[i_gr], teacher_chunk, grpo_trust_eps
                        )
                        distillation_examples_out.append(
                            observation_actions_to_bc_example(
                                obs_m_d,
                                tgt_gr,
                                model_action_horizon=int(original_model.action_horizon),
                                model_action_dim=int(original_model.action_dim),
                                alignment_ratio=float(alignment_ratio),
                                replan_env_step=int(t),
                                temporal_decay=float(temporal_decay),
                                teacher_chunk_var_mean=teacher_var_row,
                                grpo_advantage=float(adv_d[i_gr]),
                                grpo_group_weight=w_grpo,
                                rollout_trajectory_id=rollout_trajectory_id,
                            )
                        )
                else:
                    curr_obs_dict_d = _build_curr_obs_dict(
                        obs, img, wrist_img, task_description
                    )
                    policy._rng, rng_dd = jax.random.split(policy._rng)
                    noise_dd = jax.random.normal(
                        rng_dd,
                        (num_samples, original_model.action_horizon, original_model.action_dim),
                    )
                    student_stacked_dd = _infer_action_chunk_samples(
                        policy, curr_obs_dict_d, noise_dd
                    )
                    if student_stacked_dd.ndim == 3:
                        action_chunk_dd = student_stacked_dd[0]
                    else:
                        action_chunk_dd = student_stacked_dd
                    alignment_ratio_dd = compute_alignment_ratio(
                        policy,
                        action_chunk_dd,
                        curr_obs_dict_d,
                        noise=noise_dd,
                        cfg_weight=cfg_weight,
                    )
                    tgs_dd = max(1, int(teacher_group_size))
                    if tgs_dd == 1:
                        teacher_noise_dd = noise_dd[0:1]
                    else:
                        policy._rng, rng_tdd = jax.random.split(policy._rng)
                        teacher_noise_dd = jax.random.normal(
                            rng_tdd,
                            (tgs_dd, original_model.action_horizon, original_model.action_dim),
                        )
                    teacher_stacked_dd = _infer_action_chunk_samples(
                        teacher_policy, curr_obs_dict_d, teacher_noise_dd
                    )
                    if teacher_stacked_dd.ndim == 3:
                        teacher_chunk_dd = teacher_stacked_dd[0]
                    else:
                        teacher_chunk_dd = teacher_stacked_dd
                    teacher_var_row_dd = 0.0
                    if tgs_dd > 1 and teacher_stacked_dd.ndim == 3:
                        teacher_var_row_dd = float(
                            jnp.mean(jnp.var(jnp.asarray(teacher_stacked_dd), axis=0))
                        )
                    st_dd = jnp.asarray(student_stacked_dd)
                    adv_dd = _compute_grpo_advantages(st_dd, teacher_chunk_dd)
                    dists_dd = _compute_grpo_dists_np(st_dd, teacher_chunk_dd)
                    w_grpo_dd = (
                        _compute_grpo_mean_std_weight(dists_dd, eps=grpo_weight_eps)
                        if grpo_weight == "mean_std"
                        else None
                    )
                    obs_m_dd = make_observation_from_simulator(
                        policy, curr_obs_dict_d
                    )
                    for i_gr in range(int(st_dd.shape[0])):
                        tgt_gr = _grpo_trust_clamp_target_action(
                            st_dd[i_gr], teacher_chunk_dd, grpo_trust_eps
                        )
                        distillation_examples_out.append(
                            observation_actions_to_bc_example(
                                obs_m_dd,
                                tgt_gr,
                                model_action_horizon=int(original_model.action_horizon),
                                model_action_dim=int(original_model.action_dim),
                                alignment_ratio=float(alignment_ratio_dd),
                                replan_env_step=int(t),
                                temporal_decay=float(temporal_decay),
                                teacher_chunk_var_mean=teacher_var_row_dd,
                                grpo_advantage=float(adv_dd[i_gr]),
                                grpo_group_weight=w_grpo_dd,
                                rollout_trajectory_id=rollout_trajectory_id,
                            )
                        )
            elif replan_now:
                target_chunk_d = (1.0 - alpha) * teacher_chunk + alpha * action_chunk
                obs_m_d = make_observation_from_simulator(policy, curr_obs_dict)
                distillation_examples_out.append(
                    observation_actions_to_bc_example(
                        obs_m_d,
                        target_chunk_d,
                        model_action_horizon=int(original_model.action_horizon),
                        model_action_dim=int(original_model.action_dim),
                        alignment_ratio=float(alignment_ratio),
                        replan_env_step=int(t),
                        temporal_decay=float(temporal_decay),
                        teacher_chunk_var_mean=teacher_var_row,
                        rollout_trajectory_id=rollout_trajectory_id,
                    )
                )
            else:
                curr_obs_dict_d = _build_curr_obs_dict(
                    obs, img, wrist_img, task_description
                )
                policy._rng, rng_dd = jax.random.split(policy._rng)
                noise_dd = jax.random.normal(
                    rng_dd,
                    (num_samples, original_model.action_horizon, original_model.action_dim),
                )
                student_stacked_dd = _infer_action_chunk_samples(
                    policy, curr_obs_dict_d, noise_dd
                )
                if student_stacked_dd.ndim == 3:
                    action_chunk_dd = student_stacked_dd[0]
                else:
                    action_chunk_dd = student_stacked_dd
                alignment_ratio_dd = compute_alignment_ratio(
                    policy,
                    action_chunk_dd,
                    curr_obs_dict_d,
                    noise=noise_dd,
                    cfg_weight=cfg_weight,
                )
                tgs_dd = max(1, int(teacher_group_size))
                if tgs_dd == 1:
                    teacher_noise_dd = noise_dd[0:1]
                else:
                    policy._rng, rng_tdd = jax.random.split(policy._rng)
                    teacher_noise_dd = jax.random.normal(
                        rng_tdd,
                        (tgs_dd, original_model.action_horizon, original_model.action_dim),
                    )
                teacher_stacked_dd = _infer_action_chunk_samples(
                    teacher_policy, curr_obs_dict_d, teacher_noise_dd
                )
                if teacher_stacked_dd.ndim == 3:
                    teacher_chunk_dd = teacher_stacked_dd[0]
                else:
                    teacher_chunk_dd = teacher_stacked_dd
                teacher_var_row_dd = 0.0
                if tgs_dd > 1 and teacher_stacked_dd.ndim == 3:
                    teacher_var_row_dd = float(
                        jnp.mean(jnp.var(jnp.asarray(teacher_stacked_dd), axis=0))
                    )
                target_chunk_dd = (1.0 - alpha) * teacher_chunk_dd + alpha * action_chunk_dd
                obs_m_dd = make_observation_from_simulator(
                    policy, curr_obs_dict_d
                )
                distillation_examples_out.append(
                    observation_actions_to_bc_example(
                        obs_m_dd,
                        target_chunk_dd,
                        model_action_horizon=int(original_model.action_horizon),
                        model_action_dim=int(original_model.action_dim),
                        alignment_ratio=float(alignment_ratio_dd),
                        replan_env_step=int(t),
                        temporal_decay=float(temporal_decay),
                        teacher_chunk_var_mean=teacher_var_row_dd,
                        rollout_trajectory_id=rollout_trajectory_id,
                    )
                )

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
        episode_test_losses,
        neighbor_previews,
        alignment_ratio_by_step,
        ttt_count,
        policy,
        jax_key,
        teacher_l2_by_env_step,
        group_sampling_trace,
        teacher_alignment_ratio_by_step,
        teacher_group_sampling_trace,
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
    num_neighbors_fetch: int = 32,
    ttt_use_modalities: Optional[List[str]] = None,
    plot_observations: bool = False,
    repeat_batch: int = 1,
    meta_update: str = "reset",
    no_reset: bool = False,
    use_test_task: bool = False,
    random_neighbors: bool = False,
    cfg_weight: float = 1.0,
    num_samples: int = 1,
    debug_metrics: bool = False,
    merging_eps: float | None = None,
    adapt_kwargs: dict[str, Any] = {},
    disable_adaptation: bool = False,
    teacher_policy: _policy.Policy | None = None,
    show_progress_bar: bool = True,
    distillation_examples_out: list[dict[str, Any]] | None = None,
    student_action_merge: float = 0.0,
    temporal_decay: float = 1.0,
    teacher_group_size: int = 1,
    write_auxiliary_rollout_pdfs: bool = True,
    after_each_rollout_episode: Callable[[int, int, int], None] | None = None,
    grpo_like: bool = False,
    grpo_trust_eps: float | None = None,
    grpo_weight: str | None = None,
    grpo_weight_eps: float = 1e-8,
    distill_collect_every: int = 1,
    distill_trajectory_id_offset: int = 0,
    num_envs: int = 1,
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
        after_each_rollout_episode: If set, called after each trial with
            ``(episode_idx, task_episodes, task_successes)``.

    Returns:
        success_rate: Success rate across all evaluation episodes
    """
    try:
        run_evaluation.last_teacher_student_l2_pdf = None  # type: ignore[attr-defined]
        run_evaluation.last_episode_metrics = None  # type: ignore[attr-defined]
    except Exception:
        pass

    #########################################################
    # Part 1: Initialize the environment and the policy
    #########################################################
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    LIBERO_ENV_RESOLUTION = 224 if task_suite_name == "libero_90" else 256
    RESIZE_SIZE = 224
    REPLAN_STEPS = 5
    VIDEO_OUT_PATH = video_out_path
    if distillation_examples_out is not None and teacher_policy is not None:
        _grpo_msg = (
            " GRPO-like: G student chunks per replan group, advantages from -L2(chunk, teacher), "
            "train mean_i (A_i * denoise_loss_i)."
            if grpo_like
            else ""
        )
        _dce = int(distill_collect_every)
        if _dce == 1:
            _freq_msg = (
                "BC rows append every env step (after wait) via on-policy student/teacher queries."
            )
        else:
            _freq_msg = (
                f"BC rows append every {_dce} env steps (after wait; "
                f"(t-num_steps_wait) mod {_dce} == 0) via on-policy student/teacher queries."
            )
        print(
            "Distillation rollout: env still uses replan_steps=5 for executed actions; "
            + _freq_msg
            + _grpo_msg
        )

    # Seed all random number generators for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    jax_key = jax.random.PRNGKey(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if num_neighbors_fetch < 1:
        raise ValueError(f"num_neighbors_fetch must be >= 1, got {num_neighbors_fetch}")
    if ttt_k < 1:
        raise ValueError(f"ttt_k (TTT minibatch size) must be >= 1, got {ttt_k}")
    valid_meta_update = {"reset", "continual_ttt", "tt_reptile"}
    if meta_update not in valid_meta_update:
        raise ValueError(f"meta_update must be one of {sorted(valid_meta_update)}, got {meta_update}")
    if meta_update == "tt_reptile":
        if merging_eps is None:
            raise ValueError("merging_eps must be provided when meta_update='tt_reptile'")
        if not (0.0 <= merging_eps <= 1.0):
            raise ValueError(f"merging_eps must be in [0, 1], got {merging_eps}")
    elif merging_eps is not None:
        raise ValueError("merging_eps is only valid when meta_update='tt_reptile'")
    if temporal_decay < 0:
        raise ValueError(f"temporal_decay must be >= 0, got {temporal_decay}")
    if teacher_group_size < 1:
        raise ValueError(f"teacher_group_size must be >= 1, got {teacher_group_size}")
    if grpo_like:
        if int(num_samples) < 2:
            raise ValueError(
                f"grpo_like requires num_samples (group_size) >= 2, got {num_samples}"
            )
    if grpo_trust_eps is not None and not grpo_like:
        raise ValueError("grpo_trust_eps requires grpo_like=True")
    if grpo_trust_eps is not None and float(grpo_trust_eps) <= 0.0:
        raise ValueError(f"grpo_trust_eps must be > 0 when set, got {grpo_trust_eps}")
    _gw = grpo_weight if grpo_weight is not None else "none"
    if _gw not in ("none", "mean_std"):
        raise ValueError(
            f"grpo_weight must be 'none' or 'mean_std', got {grpo_weight!r}"
        )
    if _gw == "mean_std" and not grpo_like:
        raise ValueError("grpo_weight='mean_std' requires grpo_like=True")
    if float(grpo_weight_eps) <= 0.0:
        raise ValueError(f"grpo_weight_eps must be > 0, got {grpo_weight_eps}")
    if int(distill_collect_every) < 1:
        raise ValueError(
            f"distill_collect_every must be >= 1, got {distill_collect_every}"
        )
    if int(num_envs) < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")

    # Start evaluation
    task_episodes, task_successes = 0, 0
    # Collect per-episode metrics for external logging / plotting (e.g. ttt_evaluation.py).
    # We will attach this data to the function object at the end without changing
    # the public return type.
    all_episode_metrics: list[dict[str, Any]] = []
    all_teacher_l2_episodes: list[list[tuple[int, float]]] = []
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

    if int(num_envs) > 1:
        if not disable_adaptation:
            raise ValueError("num_envs > 1 requires disable_adaptation=True (no TTT during rollout)")
        if plot_observations:
            raise ValueError("num_envs > 1 requires plot_observations=False")
        if teacher_policy is None or distillation_examples_out is None:
            raise ValueError(
                "num_envs > 1 is only implemented for on-policy distillation rollouts "
                "(teacher_policy and distillation_examples_out must be set)."
            )
        original_model = policy._model
        print_trainable_parameters(nnx.state(original_model), train_config.trainable_filter)
        print(f"Task: {task.language}")
        print(
            f"Parallel distillation rollout: num_envs={num_envs} "
            "(batched student/teacher policy inference when multiple envs replan on the same step)."
        )
        from meta_libero.src.ttt.parallel_distillation_rollout import (
            run_parallel_distillation_waves,
        )

        return run_parallel_distillation_waves(
            num_envs=int(num_envs),
            policy=policy,
            teacher_policy=teacher_policy,
            original_model=original_model,
            task=task,
            task_suite_name=task_suite_name,
            task_id=task_id,
            task_description=str(task.language),
            initial_states=initial_states,
            num_trials=num_trials,
            num_steps_wait=num_steps_wait,
            max_steps=max_steps,
            seed=seed,
            save_video=save_video,
            video_out_path=VIDEO_OUT_PATH,
            show_progress_bar=show_progress_bar,
            distillation_examples_out=distillation_examples_out,
            student_action_merge=student_action_merge,
            group_size=int(num_samples),
            teacher_group_size=teacher_group_size,
            temporal_decay=temporal_decay,
            grpo_like=grpo_like,
            grpo_trust_eps=grpo_trust_eps,
            grpo_weight=grpo_weight,
            grpo_weight_eps=grpo_weight_eps,
            distill_collect_every=distill_collect_every,
            distill_trajectory_id_offset=distill_trajectory_id_offset,
            write_auxiliary_rollout_pdfs=write_auxiliary_rollout_pdfs,
            cfg_weight=cfg_weight,
            after_each_rollout_episode=after_each_rollout_episode,
        )

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

    # Run evaluation episodes; env.close() after loop — OffScreenRenderEnv EGL/MuJoCo
    # otherwise retains GPU memory across repeated run_evaluation during training.
    pbar = tqdm(
        range(num_trials),
        desc=f"Task {task_id} | Success: 0/0 (0.0%)",
        disable=not show_progress_bar,
    )
    for episode_idx in pbar:
        pbar.write(f"Episode {episode_idx+1} of {num_trials}")
        (
            done,
            t,
            replay_images,
            distances_actions,
            similarities,
            episode_losses,
            episode_test_losses,
            neighbor_previews,
            alignment_ratio_by_step,
            ttt_count,
            policy,
            jax_key,
            teacher_l2_trace,
            group_sampling_trace,
            teacher_align_trace,
            teacher_var_trace,
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
            num_neighbors_fetch=num_neighbors_fetch,
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
            merging_eps=merging_eps,
            adapt_kwargs=adapt_kwargs,
            meta_update=meta_update,
            no_reset=no_reset,
            cfg_weight=cfg_weight,
            ttt_count=ttt_count,
            jax_key=jax_key,
            pbar=pbar,
            teacher_policy=teacher_policy,
            distillation_examples_out=distillation_examples_out,
            student_action_merge=student_action_merge,
            temporal_decay=temporal_decay,
            teacher_group_size=teacher_group_size,
            grpo_like=grpo_like,
            grpo_trust_eps=grpo_trust_eps,
            grpo_weight=grpo_weight,
            grpo_weight_eps=grpo_weight_eps,
            distill_collect_every=distill_collect_every,
            rollout_trajectory_id=int(distill_trajectory_id_offset) + int(episode_idx),
        )
        all_teacher_l2_episodes.append(list(teacher_l2_trace))

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
                "test_losses": episode_test_losses,
                "num_steps": t,
                "alignment_ratio_by_step": list(alignment_ratio_by_step),
                "group_sampling_trace": list(group_sampling_trace),
                "teacher_l2_by_env_step": list(teacher_l2_trace),
                "teacher_alignment_ratio_by_step": list(teacher_align_trace),
                "teacher_group_sampling_trace": list(teacher_var_trace),
            }
        )

        if write_auxiliary_rollout_pdfs and num_samples > 1 and group_sampling_trace:
            gpdf = pathlib.Path(VIDEO_OUT_PATH) / (
                f"group_action_sampling_ep{episode_idx:03d}.pdf"
            )
            outp = _plot_group_action_sampling_pdf(
                group_sampling_trace,
                gpdf,
                episode_idx=episode_idx,
                group_size=num_samples,
            )
            if outp is not None:
                pbar.write(f"  Saved group sampling metrics plot to {outp}")

        if write_auxiliary_rollout_pdfs:
            _plot_alignment_ratio_by_step(list(alignment_ratio_by_step))
        _save_episode_losses_plot(
            episode_losses=episode_losses,
            ttt_num_steps=ttt_num_steps,
            video_out_path=VIDEO_OUT_PATH,
            episode_idx=episode_idx,
            done=done,
            filename_prefix="losses",
            y_label="Train loss",
        )
        _save_episode_losses_plot(
            episode_losses=episode_test_losses,
            ttt_num_steps=ttt_num_steps,
            video_out_path=VIDEO_OUT_PATH,
            episode_idx=episode_idx,
            done=done,
            filename_prefix="test_losses",
            y_label="Test loss",
        )
        if plot_observations:
            _save_episode_neighbors_plot(
                neighbor_previews=neighbor_previews,
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

        del neighbor_previews, replay_images

        if after_each_rollout_episode is not None:
            after_each_rollout_episode(episode_idx, task_episodes, task_successes)

        # Log progress
        if (episode_idx + 1) % 1 == 0:
            pbar.write(
                f"  Episodes: {task_episodes}, Successes: {task_successes} ({task_successes/task_episodes*100:.1f}%)"
            )

    try:
        env.close()
    except Exception:
        pass

    if distillation_examples_out is not None:
        n_d = len(distillation_examples_out)
        _dce = int(distill_collect_every)
        print(
            f"Distillation: collected {n_d} trajectory samples this evaluation run "
            f"({num_trials} episode(s); BC rows when "
            f"(t-num_steps_wait) mod {_dce} == 0 after wait, teacher set)."
        )

    teacher_l2_pdf: pathlib.Path | None = None
    if teacher_policy is not None and write_auxiliary_rollout_pdfs:
        pdf_path = pathlib.Path(VIDEO_OUT_PATH) / (
            f"teacher_student_action_l2_task{task_id}.pdf"
        )
        teacher_l2_pdf = _plot_teacher_student_action_l2_pdf(
            all_teacher_l2_episodes,
            pdf_path,
            task_id=task_id,
        )
        if teacher_l2_pdf is not None:
            print(f"Saved teacher vs student action L2 plot to {teacher_l2_pdf}")
        try:
            run_evaluation.last_teacher_student_l2_pdf = (  # type: ignore[attr-defined]
                str(teacher_l2_pdf) if teacher_l2_pdf is not None else None
            )
        except Exception:
            pass
    elif teacher_policy is not None:
        try:
            run_evaluation.last_teacher_student_l2_pdf = None  # type: ignore[attr-defined]
        except Exception:
            pass

    # Expose collected metrics on the function object without changing the
    # public return type, so scripts can read them after a call.
    try:
        run_evaluation_ttt.last_episode_metrics = all_episode_metrics  # type: ignore[attr-defined]
        run_evaluation.last_episode_metrics = all_episode_metrics  # type: ignore[attr-defined]
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
    ttt_training_data: List[Any],
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
    batch_size = kwargs["batch_size"]

    ttt_data_loader = NeighborsDataLoader(
        examples=ttt_training_data,
        batch_size=batch_size,
        data_config=ttt_data_config,
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
        show_progress_bar=False,
        resume_train_state=None,  # Each TTT starts fresh - don't accumulate optimizer state
        resume_losses=None,  # Don't carry over losses
        donate_buffers=True,  # Enable donation to save memory (copy is modified, original preserved)
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
    try:
        adapt_fn_ttt.last_test_losses = list(aux_losses)  # type: ignore[attr-defined]
    except Exception:
        pass

    del train_state, ttt_data_loader

    return trained_model, list(train_losses)


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

    try:
        adapt_fn_gaussian_perturbation.last_test_losses = []  # type: ignore[attr-defined]
    except Exception:
        pass
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
