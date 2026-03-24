"""General utility helpers."""

from __future__ import annotations

import dataclasses
import math
import pathlib
import time
from typing import Any, Optional, Tuple, List

import jax
import jax.numpy as jnp
import numpy as np
import torch
import openpi.models.pi0_config as pi0_config
from tqdm import tqdm

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import openpi.models.model as _model
from openpi.policies import policy as _policy
from openpi.policies import libero_policy as _openpi_libero_policy
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.shared.download as download
from openpi.training.data_loader import Dataset
import openpi.transforms as transforms
from openpi.training import checkpoints as _checkpoints

from meta_libero.src.dataset import FilteredDataset
from meta_libero.src.libero_inputs_override import BatchableLiberoInputs, BatchableLiberoOutputs
from meta_libero.src.rendering import plot_observation_with_decoded_prompt


PRINT_MEMORY_CHECKPOINT = False


def get_gpu_memory_usage() -> dict[str, float] | dict[str, str] | None:
    """Get current GPU memory usage in GB."""
    try:
        devices = jax.devices()
        if not devices:
            return None
        mem_info = devices[0].memory_stats()
        return {
            "allocated_gb": mem_info.get("bytes_in_use", 0) / (1024**3),
            "reserved_gb": mem_info.get("bytes_reserved", 0) / (1024**3),
            "total_gb": mem_info.get("bytes_reserved", 0) / (1024**3),
        }
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}


def print_memory_checkpoint(label: str, line_num: int | None = None) -> None:
    """Print memory usage at a checkpoint."""
    if not PRINT_MEMORY_CHECKPOINT:
        return
    mem = get_gpu_memory_usage()
    if isinstance(mem, dict) and "error" not in mem:
        line_info = f" (line {line_num})" if line_num else ""
        print(
            f"[MEMORY CHECKPOINT{line_info}] {label}: "
            f"Allocated: {mem['allocated_gb']:.2f} GB, "
            f"Reserved: {mem['reserved_gb']:.2f} GB"
        )
    else:
        print(f"[MEMORY CHECKPOINT] {label}: {mem}")



def fetch_samples(dataset, idx, repeat=1) -> List[Any]:
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
        # NN indices are physical/global; map them to logical indices within FilteredDataset.
        physical_to_logical = {int(physical_idx): logical_idx for logical_idx, physical_idx in enumerate(filtered_ds.indices)}
        mapped = [physical_to_logical[i] for i in idx_int]
        examples = [dataset[mapped_i] for mapped_i in mapped]
    else:
        examples = [dataset[i] for i in idx_int]

    examples = [example for _ in range(repeat) for example in examples]
    return examples



def load_pi05_libero_model(
    use_base_model: bool = False,
    use_lora: bool = False,
    action_expert_only: bool = False
) -> Tuple[_model.BaseModel, _config.TrainConfig]:
    if use_base_model:
        checkpoint_gs_path = "gs://openpi-assets/checkpoints/pi05_base"
    else:
        checkpoint_gs_path = "gs://openpi-assets/checkpoints/pi05_libero"

    checkpoint_name = "pi05_libero" if not use_lora else "pi05_libero_lora"
    config = _config.get_config(checkpoint_name)

    checkpoint_dir = download.maybe_download(checkpoint_gs_path)
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


def new_policy_like(policy: _policy.Policy, model: _model.BaseModel):
    return _policy.Policy(
        model,
        transforms=policy._input_transform.transforms,
        output_transforms=policy._output_transform.transforms,
        sample_kwargs=policy._sample_kwargs,
        metadata=policy._metadata,
        is_pytorch=policy._is_pytorch_model,
        pytorch_device=policy._pytorch_device,
    )


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
    repack_transforms = repack_transforms or transforms.Group()
    libero_inputs_cls = _openpi_libero_policy.LiberoInputs
    libero_outputs_cls = _openpi_libero_policy.LiberoOutputs
    if _openpi_libero_policy.LiberoInputs is not BatchableLiberoInputs:
        _openpi_libero_policy.LiberoInputs = BatchableLiberoInputs
    if _openpi_libero_policy.LiberoOutputs is not BatchableLiberoOutputs:
        _openpi_libero_policy.LiberoOutputs = BatchableLiberoOutputs

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

    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(
            pathlib.Path(checkpoint_dir) / "assets", data_config.asset_id
        )

    _ = jax.random.PRNGKey(rng_seed) if rng_seed is not None else None

    return _policy.Policy(
        model,
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


def nn_lookup(
    observation: _model.Observation,
    nn_fetcher,
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

    if nn_fetcher.normalize_per_modality:  # True for libero90, False for libero10
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

    examples = fetch_samples(dataset, indices, repeat=repeat_batch)

    if use_test_task:
        raise NotImplementedError("Test task needs to be reimplemented after refactoring the data loader")
        nn_observations = dataclasses.replace(
            nn_observations,
            tokenized_prompt=jnp.repeat(observation.tokenized_prompt, k, axis=0),
            tokenized_prompt_mask=jnp.repeat(observation.tokenized_prompt_mask, k, axis=0),
            token_ar_mask=jnp.repeat(observation.token_ar_mask, k, axis=0) if observation.token_ar_mask is not None else None,
            token_loss_mask=jnp.repeat(observation.token_loss_mask, k, axis=0) if observation.token_loss_mask is not None else None,
        )

    if plot_observations:

        batch = _data_loader._collate_fn(examples)

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
        nn_observations = _model.Observation.from_dict(batch)
        
        plot_observation_with_decoded_prompt(
            observation=observation,
            similarity_score=None,
            plot_title_prefix=f"\tCurrent Observation",
        )
        if len(indices) > 0 and len(distances) > 0:
            plot_observation_with_decoded_prompt(
                observation=nn_observations,
                similarity_score=distances,
                plot_title_prefix=f"\tFetched Observation (Best Match)",
            )
    return examples, distances


def _compute_aux_denoising_losses(
    model: _model.BaseModel,
    observation: _model.Observation,
    actions: _model.Actions,
    *,
    seed: int,
    num_samples: int,
) -> list[float]:
    losses: list[float] = []
    for i in range(num_samples):
        rng = jax.random.PRNGKey(seed + i)
        chunked_loss = model.compute_loss(rng, observation, actions, train=False)
        losses.append(float(jax.device_get(jnp.mean(chunked_loss))))
    return losses


def compute_alignment_ratio(
    policy: _policy.Policy,
    action_chunk: _model.Actions,
    curr_obs_dict: dict[str, Any],
    noise: np.ndarray | jnp.ndarray | None = None,
    cfg_weight: float = 1.0,
    return_per_sample: bool = False,
) -> float | jnp.ndarray:
    """Compute alignment ratio: ||action - action_empty|| / (||action_empty|| + eps).

    Norm is taken over axes 1 and 2 (action horizon, action dim). Batched: returns shape (B,).

    Args:
        policy: Policy used for inference.
        action_chunk: Actions to compare (e.g. reference model output).
        curr_obs_dict: Observation dict (modified in-place for empty prompt).
        noise: Optional noise for policy inference.
        cfg_weight: Unused, kept for API compatibility.
        return_per_sample: If True, return per-sample ratios (shape (N,)); else return scalar mean.

    Returns:
        Scalar mean alignment ratio, or per-sample ratios when return_per_sample=True.
    """
    del cfg_weight  # unused
    original_prompt = curr_obs_dict["prompt"]
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

    s = min(action_chunk_jax.shape[0], action_chunk_empty_jax.shape[0])
    h = min(action_chunk_jax.shape[1], action_chunk_empty_jax.shape[1], 5)
    d = min(action_chunk_jax.shape[2], action_chunk_empty_jax.shape[2])
    action_chunk_overlap = action_chunk_jax[:s, :h, :d]
    action_chunk_empty_overlap = action_chunk_empty_jax[:s, :h, :d]

    # Norm over axes 1 and 2, then ratio -> shape (B,)
    diff = action_chunk_overlap - action_chunk_empty_overlap
    num = jnp.linalg.norm(diff, axis=(1, 2))
    den = jnp.linalg.norm(action_chunk_empty_overlap, axis=(1, 2))
    ratios = num / (den + 1e-12)

    if return_per_sample:
        return ratios
    return float(jax.device_get(jnp.mean(ratios)))

__all__ = [
    "PRINT_MEMORY_CHECKPOINT",
    "get_gpu_memory_usage",
    "print_memory_checkpoint",
    "fetch_samples",
    "load_pi05_libero_model",
    "_quat2axisangle",
    "_get_libero_env",
    "new_policy_like",
    "create_policy",
    "nn_lookup",
    "_compute_aux_denoising_losses",
    "compute_alignment_ratio",
]

