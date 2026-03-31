"""Canonical dataset utilities for meta_libero."""

import collections
import contextlib
import dataclasses
import logging
import os
from pathlib import Path
from collections.abc import Callable
from typing import Any, Dict, List, Optional

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import torch
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # type: ignore
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
import openpi.models.model as _model  # type: ignore
from openpi.training import config as _config  # type: ignore
import openpi.training.data_loader as _data_loader  # type: ignore
from openpi.training.data_loader import (  # type: ignore
    Dataset,
    DataLoaderImpl,
    TorchDataLoader,
    TransformedDataset,
    transform_dataset,
)
import openpi.transforms as transforms  # type: ignore
from openpi_client import image_tools  # type: ignore


_TASK_INDICES_CACHE: dict[str, dict[int, list[int]]] = {}

# Raw LIBERO low-dim proprio length. Training pads to ``model.action_dim`` (e.g. 32) via
# ``PadStatesAndActions``; policy ``infer`` must receive unpadded state so quantile norm matches stats.
_LIBERO_STATE_DIM = 8

LIBERO_90_TASK_IDS_PROMPTS = {
    0: "close the top drawer of the cabinet",
    1: "close the top drawer of the cabinet and put the black bowl on top of it",
    2: "put the black bowl in the top drawer of the cabinet",
    3: "put the butter at the back in the top drawer of the cabinet and close it",
    4: "put the butter at the front in the top drawer of the cabinet and close it",
    5: "put the chocolate pudding in the top drawer of the cabinet and close it",
    6: "open the bottom drawer of the cabinet",
    7: "open the top drawer of the cabinet",
    8: "open the top drawer of the cabinet and put the bowl in it",
}

LIBERO_90_TASK_IDS_MAPPING = {
    0: 55,
    1: 63,
    2: 11,
    3: 10,
    4: 19,
    5: 9,
    6: 59,
    7: 24,
    8: 3,
}

# Map simulator task_id -> episode_index to use when single-episode mode is enabled.
# Populate this as needed (kept empty by default).
SINGLE_EPISODE_TASK_TO_EPISODE_INDEX: dict[int, int] = {
    0: 98,
    1: 129,
    2: 260,
    3: 12,
    4: 23,
    5: 254,
    6: 118,
    7: 28,
    8: 4,
}


def _build_task_indices_map(dataset: Dataset) -> dict[int, list[int]]:
    """Build a map from task_id/task_index to dataset sample indices."""
    task_to_indices: dict[int, list[int]] = collections.defaultdict(list)
    if hasattr(dataset, "hf_dataset") and "task_index" in dataset.hf_dataset.column_names:
        task_indices = dataset.hf_dataset["task_index"]
        for idx, task_idx in enumerate(task_indices):
            task_to_indices[int(task_idx)].append(idx)
    else:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if "task_index" in sample:
                task_to_indices[int(sample["task_index"])].append(idx)
    return dict(task_to_indices)


def _get_cached_task_indices(repo_id: str, dataset: Dataset) -> dict[int, list[int]]:
    """Return cached task->indices mapping for this repo, building once if needed."""
    if repo_id not in _TASK_INDICES_CACHE:
        print(f"Building task index cache for repo '{repo_id}'...")
        _TASK_INDICES_CACHE[repo_id] = _build_task_indices_map(dataset)
        print(f"Task index cache built with {len(_TASK_INDICES_CACHE[repo_id])} tasks")
    return _TASK_INDICES_CACHE[repo_id]


class FilteredDataset(Dataset):
    """Wraps a dataset and filters samples by task_index/task_id."""

    def __init__(
        self,
        dataset: Dataset,
        task_index: int,
        repo_id: str | None = None,
        episode_index: int | None = None,
    ):
        self.dataset = dataset
        self.task_index = task_index
        self.episode_index = episode_index

        print(f"Filtering dataset for task_index={task_index} and episode_index={episode_index}...")
        if repo_id is not None:
            task_to_indices = _get_cached_task_indices(repo_id, dataset)
            self.indices = list(task_to_indices.get(int(task_index), []))
            print(f"Filtered dataset (cached): {len(self.indices)} / {len(dataset)} samples")
        else:
            self.indices = _build_task_indices_map(dataset).get(int(task_index), [])
            print(f"Filtered dataset (uncached): {len(self.indices)} / {len(dataset)} samples")

        if episode_index is not None:
            self.indices = self._filter_indices_by_episode(self.indices, int(episode_index))
            print(f"Episode-filtered: {len(self.indices)} samples for episode_index={episode_index}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def _filter_indices_by_episode(self, indices: list[int], episode_index: int) -> list[int]:
        """Filter physical dataset indices by episode index."""
        if hasattr(self.dataset, "hf_dataset") and "episode_index" in self.dataset.hf_dataset.column_names:
            episode_col = self.dataset.hf_dataset["episode_index"]
            return [i for i in indices if int(episode_col[i]) == episode_index]

        filtered: list[int] = []
        for i in indices:
            sample = self.dataset[i]
            if "episode_index" in sample and int(sample["episode_index"]) == episode_index:
                filtered.append(i)
        return filtered


def _data_to_policy_obs_dict(data: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Convert dataset sample (post data_transforms) to policy input format.

    Data has image (dict with base_0_rgb, left_wrist_0_rgb), state. Policy expects
    observation/image, observation/wrist_image, observation/state, prompt.
    """
    img = data.get("image")
    if isinstance(img, dict):
        base_img = np.asarray(img.get("base_0_rgb", np.zeros((224, 224, 3), dtype=np.uint8)))
        wrist_img = np.asarray(img.get("left_wrist_0_rgb", np.zeros((224, 224, 3), dtype=np.uint8)))
    elif "observation/image" in data:
        base_img = np.asarray(data["observation/image"])
        wrist_img = np.asarray(data.get("observation/wrist_image", np.zeros((224, 224, 3), dtype=np.uint8)))
    else:
        base_img = np.asarray(img) if img is not None else np.zeros((224, 224, 3), dtype=np.uint8)
        wrist_img = np.zeros((224, 224, 3), dtype=np.uint8)
    state = np.asarray(
        data.get("state", data.get("observation/state", np.zeros(_LIBERO_STATE_DIM, dtype=np.float32)))
    )
    if state.shape[-1] > _LIBERO_STATE_DIM:
        state = state[..., :_LIBERO_STATE_DIM]
    return {
        "observation/image": base_img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": prompt,
    }


@dataclasses.dataclass(frozen=True)
class MakePseudoLabelTransform(transforms.DataTransformFn):
    """Transform that injects prompt and actions before model_transforms.

    By default: prompt='do nothing', actions=zeros.
    If inference_fn is provided: uses inference_fn(data) to get pseudo-label actions.
    Insert this before TokenizePrompt.
    """

    prompt: str = "do nothing"
    inference_fn: Callable[[dict[str, Any]], np.ndarray] | None = None

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        out = dict(data)
        out["prompt"] = self.prompt
        if self.inference_fn is not None:
            out["actions"] = np.asarray(self.inference_fn(out))
        elif "actions" in out:
            out["actions"] = np.zeros_like(np.asarray(out["actions"]))
        return out


def make_pseudo_label_inference_fn(policy: Any) -> Callable[[dict[str, Any]], np.ndarray]:
    """Create inference_fn for MakePseudoLabelTransform from a policy.

    The returned callable receives a data dict (after prompt is set by the transform),
    builds policy obs format, runs policy.infer, and returns actions for one sample.
    """
    def fn(data: dict[str, Any]) -> np.ndarray:
        p = data["prompt"]
        obs = _data_to_policy_obs_dict(data, p)
        result = policy.infer(obs, noise=None)
        actions = np.asarray(result["actions"])
        if actions.ndim == 3:
            actions = actions[0]
        return actions
    return fn


def augment_dataset_with_pseudo_labels(
    dataset: Dataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    prompt: str = "do nothing",
    inference_fn: Callable[[dict[str, Any]], np.ndarray] | None = None,
) -> TransformedDataset:
    """Like transform_dataset but injects prompt and actions before model_transforms.

    Produces samples with the given prompt and either zero actions or pseudo-labels
    from inference_fn. Same pipeline as transform_dataset except model_transforms
    receive the injected prompt/actions.
    """
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                f"Normalization stats not found for repo {data_config.repo_id}. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    inject = MakePseudoLabelTransform(prompt=prompt, inference_fn=inference_fn)
    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            inject,
            *data_config.model_transforms.inputs,
        ],
    )


class PairDataset(Dataset):
    """Unified dataset that yields (sample1, sample2) pairs for batch training.

    Supports two sampling modes:
    - paired=True (self_replay): same index for both, (dataset1[i], dataset2[i])
    - paired=False (cotraining/on_policy): independent sampling,
      (dataset1[perm1[i]], dataset2[perm2[i]])
    Used with _collate_pair_fn to produce batches of 2*B samples.
    """

    def __init__(
        self,
        dataset1: Dataset,
        dataset2: Dataset,
        *,
        paired: bool = True,
        seed: int = 0,
    ):
        self._d1 = dataset1
        self._d2 = dataset2
        self._paired = paired
        if paired:
            assert len(dataset1) == len(dataset2), "paired mode requires same length"
            self._len = len(dataset1)
        else:
            n1, n2 = len(dataset1), len(dataset2)
            self._len = max(n1, n2)
            rng = np.random.default_rng(seed)
            self._perm1 = rng.permutation(n1)
            self._perm2 = rng.permutation(n2)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._paired:
            s1 = self._d1[idx]
            s2 = self._d2[idx]
        else:
            n1, n2 = len(self._d1), len(self._d2)
            s1 = self._d1[int(self._perm1[idx % n1])]
            s2 = self._d2[int(self._perm2[idx % n2])]
        return jax.tree.map(
            lambda a, b: np.concatenate([np.asarray(a)[None, ...], np.asarray(b)[None, ...]], axis=0),
            s1,
            s2,
        )


def _collate_pair_fn(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate items from PairDataset: stack (B, 2, ...) then reshape to (2B, ...)."""
    stacked = jax.tree.map(
        lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0),
        *items,
    )
    return jax.tree.map(
        lambda x: np.reshape(np.asarray(x), (-1,) + x.shape[2:]),
        stacked,
    )




class _FlexibleTorchDataLoader:
    """TorchDataLoader with configurable collate_fn. Mirrors openpi TorchDataLoader."""

    def __init__(
        self,
        dataset: Any,
        local_batch_size: int,
        *,
        collate_fn=None,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        import multiprocessing
        import typing

        if collate_fn is None:
            collate_fn = _data_loader._collate_fn
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")
        if len(dataset) < local_batch_size:
            raise ValueError(
                f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)})."
            )

        self._sharding = sharding
        if sharding is None and framework == "jax":
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn,
            worker_init_fn=_data_loader._worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                if self._sharding is not None:
                    yield jax.tree.map(
                        lambda x: jax.make_array_from_process_local_data(self._sharding, x),
                        batch,
                    )
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    augment: bool = False,
    augment_task_id: int | None = None,
    augment_inference_fn: Callable[[dict[str, Any]], np.ndarray] | None = None,
    single_epoch: bool = False,
    cotraining: bool = False,
    cotraining_dataset: Dataset | None = None,
    on_policy_cotraining: bool = False,
    on_policy_cotraining_dataset: Dataset | None = None,
    on_policy_inference_fn: Callable[[dict[str, Any]], np.ndarray] | None = None,
    on_policy_prompt: str = "do nothing",
):
    """Create a data loader for training.

    Entry point mirroring openpi.training.data_loader.create_torch_data_loader.
    Uses transform_dataset (which applies data_config.model_transforms.inputs).
    When augment=True: creates two datasets (normal + augmented), samples from both
    for each batch, and concatenates (batch size becomes 2x).
    When cotraining=True: combines base_dataset with cotraining_dataset, sampling
    independently from each (batch size becomes 2x).
    When on_policy_cotraining=True: like cotraining but cotraining dataset uses
    pseudo-labels (inputs from dataset, actions from on_policy_inference_fn).
    Options:
      - augment_task_id is None: "do nothing" prompt + zero actions
      - augment_task_id is set: use that task's prompt; augment_inference_fn for
        pseudo-labels (if None, actions=zeros).
    """

    # It will use the overridden dataset
    base_dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)

    if on_policy_cotraining and on_policy_cotraining_dataset is not None:
        if on_policy_inference_fn is not None and num_workers > 0:
            num_workers = 0
        dataset1 = transform_dataset(base_dataset, data_config, skip_norm_stats=skip_norm_stats)
        dataset2 = augment_dataset_with_pseudo_labels(
            on_policy_cotraining_dataset,
            data_config,
            skip_norm_stats=skip_norm_stats,
            prompt=on_policy_prompt,
            inference_fn=on_policy_inference_fn,
        )
        dataset = PairDataset(dataset1, dataset2, paired=False, seed=seed)
        local_batch_size = batch_size
    elif cotraining and cotraining_dataset is not None:
        dataset1 = transform_dataset(base_dataset, data_config, skip_norm_stats=skip_norm_stats)
        dataset2 = transform_dataset(cotraining_dataset, data_config, skip_norm_stats=skip_norm_stats)
        dataset = PairDataset(dataset1, dataset2, paired=False, seed=seed)
        local_batch_size = batch_size
    elif augment:
        # augment_inference_fn is a closure (from make_pseudo_label_inference_fn) that
        # captures the policy; it cannot be pickled for DataLoader multiprocessing.
        if augment_inference_fn is not None and num_workers > 0:
            num_workers = 0
        augment_prompt = (
            "do nothing" if augment_task_id is None
            else LIBERO_90_TASK_IDS_PROMPTS.get(augment_task_id, "do nothing")
        )
        dataset_normal = transform_dataset(base_dataset, data_config, skip_norm_stats=skip_norm_stats)
        dataset_augmented = augment_dataset_with_pseudo_labels(
            base_dataset,
            data_config,
            skip_norm_stats=skip_norm_stats,
            prompt=augment_prompt,
            inference_fn=augment_inference_fn,
        )
        dataset = PairDataset(dataset_normal, dataset_augmented, paired=True)
        # Each "sample" is a (normal, null) pair -> 2 samples. Use half batch size for pairs.
        local_batch_size = batch_size  #  // 2
    else:
        dataset = transform_dataset(base_dataset, data_config, skip_norm_stats=skip_norm_stats)
        local_batch_size = batch_size

    use_pair_collate = augment or (cotraining and cotraining_dataset is not None) or (on_policy_cotraining and on_policy_cotraining_dataset is not None)
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = local_batch_size // torch.distributed.get_world_size()
        else:
            pass  # local_batch_size already set
    else:
        local_batch_size = local_batch_size // jax.process_count()

    if single_epoch:
        num_batches = len(dataset) // local_batch_size

    if use_pair_collate:
        mode = "on_policy_cotraining" if on_policy_cotraining else ("cotraining" if cotraining else "augmented")
        logging.info(
            f"local_batch_size: {local_batch_size} pairs -> {2 * local_batch_size} samples ({mode})"
        )
        data_loader = _FlexibleTorchDataLoader(
            dataset,
            local_batch_size=local_batch_size,
            collate_fn=_collate_pair_fn,
            sharding=None if framework == "pytorch" else sharding,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_batches=num_batches,
            num_workers=num_workers,
            seed=seed,
            framework=framework,
        )
    else:
        logging.info(f"local_batch_size: {local_batch_size}")
        data_loader = TorchDataLoader(
            dataset,
            local_batch_size=local_batch_size,
            sharding=None if framework == "pytorch" else sharding,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_batches=num_batches,
            num_workers=num_workers,
            seed=seed,
            framework=framework,
        )

    return DataLoaderImpl(data_config, data_loader)


@contextlib.contextmanager
def override_create_torch_dataset(
    repo_id: Optional[str] = None,
    task_id: int | None = None,
    load_in_memory: bool = False,
    mirror_data: bool = True,
    single_episode: bool = False,
    episode_index: int | None = None,
    augment: bool = False,
    augment_task_id: int | None = None,
    augment_inference_fn: Callable[[dict[str, Any]], np.ndarray] | None = None,
    cotraining: bool = False,
    cotraining_task_id: int | None = None,
    on_policy_self_replay: bool = False,
    on_policy_cotraining_task_id: int | None = None,
    on_policy_inference_fn: Callable[[dict[str, Any]], np.ndarray] | None = None,
):
    """Temporarily override OpenPI create_torch_dataset with LIBERO-aware one.

    If augment is True, also overrides create_torch_data_loader to use our
    create_torch_data_loader, concatenating each batch with an augmented copy.
    If cotraining is True, samples batches from task_id and cotraining_task_id
    independently and concatenates (no pseudo-labels).
    If on_policy_self_replay is True: cotraining where cotraining task uses
    inputs from dataset but actions from on_policy_inference_fn (reference model).
    Options: augment_task_id=None for "do nothing"+zeros; augment_task_id set
    for task prompt + augment_inference_fn (pseudo-labels).
    """
    original_create_dataset = _data_loader.create_torch_dataset
    original_create_torch_data_loader = _data_loader.create_torch_data_loader

    def create_dataset_impl(
        task_id_to_use: int | None,
        data_config: _config.DataConfig,
        action_horizon: int,
        model_config: Any,
    ) -> Dataset:
        del model_config
        repo = repo_id if repo_id is not None else data_config.repo_id

        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo)
        dataset = lerobot_dataset.LeRobotDataset(
            repo,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )

        if load_in_memory:
            print(f"Loading dataset '{repo}' into memory...")
            if hasattr(dataset, "hf_dataset"):
                dataset.hf_dataset = dataset.hf_dataset.with_format(None)
                dataset.hf_dataset.set_format(type=None)
                _ = dataset.hf_dataset[:]
                print(f"Dataset loaded into memory: {len(dataset)} samples")

        if task_id_to_use is not None:
            if repo_id == "antoniomari/libero_90":
                task_id_dataset = LIBERO_90_TASK_IDS_MAPPING.get(task_id_to_use, task_id_to_use)
            else:
                raise NotImplementedError(f"Only supported for antoniomari/libero_90, got {repo_id}")
            ep_idx = episode_index
            if ep_idx is None and single_episode:
                ep_idx = int(SINGLE_EPISODE_TASK_TO_EPISODE_INDEX.get(task_id_to_use, 0))
            dataset = FilteredDataset(
                dataset,
                task_id_dataset,
                repo_id=repo,
                episode_index=ep_idx,
            )

        if data_config.prompt_from_task:
            dataset = TransformedDataset(dataset, [transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

        # Apply LIBERO-90 mirroring directly in the loader override when enabled.
        if mirror_data and repo == "antoniomari/libero_90":
            dataset = TransformedDataset(dataset, [Libero90MirrorTransform()])

        return dataset

    # Creates the main dataset with the given task_id
    def create_dataset(
        data_config: _config.DataConfig, action_horizon: int, model_config: Any
    ) -> Dataset:
        return create_dataset_impl(task_id, data_config, action_horizon, model_config)

    def create_loader_with_augment(*args, **kwargs):
        return create_torch_data_loader(
            *args,
            augment=True,
            augment_task_id=augment_task_id,
            augment_inference_fn=augment_inference_fn,
            **kwargs,
        )

    def create_loader_with_cotraining(*args, **kwargs):
        # Create cotraining dataset for cotraining_task_id
        data_config = args[0] if args else kwargs["data_config"]
        action_horizon = kwargs.get("action_horizon")
        model_config = kwargs.get("model_config")
        cotraining_ds = create_dataset_impl(
            cotraining_task_id, data_config, action_horizon, model_config
        )
        return create_torch_data_loader(
            *args,
            **{**kwargs, "cotraining": True, "cotraining_dataset": cotraining_ds},
        )

    def create_loader_with_on_policy_self_replay(*args, **kwargs):
        data_config = args[0] if args else kwargs["data_config"]
        action_horizon = kwargs.get("action_horizon")
        model_config = kwargs.get("model_config")
        on_policy_ds = create_dataset_impl(
            on_policy_cotraining_task_id, data_config, action_horizon, model_config
        )
        on_policy_prompt = LIBERO_90_TASK_IDS_PROMPTS.get(
            on_policy_cotraining_task_id, "do nothing"
        )
        return create_torch_data_loader(
            *args,
            **{
                **kwargs,
                "on_policy_cotraining": True,
                "on_policy_cotraining_dataset": on_policy_ds,
                "on_policy_inference_fn": on_policy_inference_fn,
                "on_policy_prompt": on_policy_prompt,
            },
        )

    # Create the main dataset with the given task_id
    _data_loader.create_torch_dataset = create_dataset
    if augment:
        _data_loader.create_torch_data_loader = create_loader_with_augment
    elif on_policy_self_replay and on_policy_cotraining_task_id is not None and on_policy_inference_fn is not None:
        _data_loader.create_torch_data_loader = create_loader_with_on_policy_self_replay
    elif cotraining and cotraining_task_id is not None:
        _data_loader.create_torch_data_loader = create_loader_with_cotraining
    try:
        yield
    finally:
        _data_loader.create_torch_dataset = original_create_dataset
        if augment or cotraining or on_policy_self_replay:
            _data_loader.create_torch_data_loader = original_create_torch_data_loader


def extract_prompt_from_filename(name: str) -> str:
    """Extract task prompt from LIBERO HDF5 filename."""
    import re

    assert name.endswith("_demo.hdf5")
    prompt = re.sub(r"^[A-Z_]+_SCENE\d+_", "", name)
    prompt = re.sub(r"_demo\.hdf5$", "", prompt).replace("_", " ")
    return prompt


def prepare_task_dataset(task_suite_name: str = "libero_90", task_id: int = 0) -> List[List[Dict]]:
    """Convert LIBERO HDF5 task to list-of-episodes model format."""
    dataset_dir = os.getenv("LIBERO_DATASET_DIR", str(Path.home() / "libero_datasets"))
    image_resolution = (256, 256)
    libero_path = Path(dataset_dir) / task_suite_name
    hdf5_files = sorted(list(libero_path.glob("*.hdf5")))

    if not hdf5_files:
        raise ValueError("No HDF5 files found in libero_90!")
    if task_id >= len(hdf5_files):
        raise ValueError(f"Task ID {task_id} out of range. Only {len(hdf5_files)} tasks available.")

    task_file = hdf5_files[task_id]
    print(f"\nLoading task {task_id}: {task_file.name}")
    prompt = extract_prompt_from_filename(task_file.name)

    episodes = []
    with h5py.File(task_file, "r") as f:
        data_group: Any = f["data"]
        demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]
        for demo_key in demo_keys:
            demo_data: Any = data_group[demo_key]
            episode = {
                "observations": {},
                "actions": np.array(demo_data["actions"]),
                "prompt": prompt,
                "state": np.concatenate(
                    [
                        np.array(demo_data["obs"]["ee_pos"]),
                        np.array(demo_data["obs"]["ee_ori"]),
                        np.array(demo_data["obs"]["gripper_states"]),
                    ],
                    axis=1,
                ),
            }
            for camera_view in ["agentview_rgb", "eye_in_hand_rgb"]:
                if camera_view in demo_data["obs"]:
                    episode["observations"][camera_view] = np.array(demo_data["obs"][camera_view])
            episodes.append(episode)

    print(f"Loaded {len(episodes)} episodes from task {task_id}")

    dataset = []
    for episode in episodes:
        episode_data = []
        for t in range(len(episode["actions"])):
            image = episode["observations"]["agentview_rgb"][t]
            image = image[::-1, ::-1]
            image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(image, image_resolution[0], image_resolution[1])
            )
            wrist_image = episode["observations"]["eye_in_hand_rgb"][t]
            wrist_image = wrist_image[::-1, ::-1]
            wrist_image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_image, image_resolution[0], image_resolution[1])
            )
            episode_data.append(
                {
                    "image": image,
                    "wrist_image": wrist_image,
                    "state": episode["state"][t],
                    "actions": episode["actions"][t],
                    "prompt": episode["prompt"],
                }
            )
        dataset.append(episode_data)

    return dataset


def convert_to_lerobot_dataset(dataset: List[List[Dict]], repo_id: str = "example") -> LeRobotDataset:
    """Convert in-memory LIBERO episode data to a LeRobotDataset."""
    lerobot_ds = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,
        features={
            "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for episode in dataset:
        for sample in episode:
            lerobot_ds.add_frame(
                {
                    "image": sample["image"],
                    "wrist_image": sample["wrist_image"],
                    "state": sample["state"].astype(np.float32),
                    "actions": sample["actions"].astype(np.float32),
                    "task": sample["prompt"],
                }
            )
        lerobot_ds.save_episode()

    return lerobot_ds


def mirror_images_jnp(img):
    """Horizontally flip images across common tensor layouts.

    - HWC / BHWC: flip width axis `-2`
    - CHW / BCHW: flip width axis `-1`
    """
    arr = jnp.asarray(img)
    if arr.ndim == 4:
        # BHWC
        if arr.shape[-1] in (1, 3, 4):
            return arr[:, :, ::-1, :]
        # BCHW
        if arr.shape[1] in (1, 3, 4):
            return arr[:, :, :, ::-1]
    if arr.ndim == 3:
        # HWC
        if arr.shape[-1] in (1, 3, 4):
            return arr[:, ::-1, :]
        # CHW
        if arr.shape[0] in (1, 3, 4):
            return arr[:, :, ::-1]
    if arr.ndim < 2:
        return arr
    return jnp.flip(arr, axis=-2)


def mirror_action_chunks_jnp(action_chunk):
    """Mirror action chunks by sign-flipping dims 1, 3, 5."""
    arr = jnp.asarray(action_chunk)
    mask = jnp.ones(arr.shape[-1], dtype=arr.dtype)
    for idx in (1, 3, 5):
        if idx < arr.shape[-1]:
            mask = mask.at[idx].set(-1.0)
    return arr * mask


def _mirror_libero90_example(data: dict[str, Any]) -> dict[str, Any]:
    """Mirror image/action fields for LIBERO-90 samples."""
    out = dict(data)

    for key in ("observation/image", "observation/wrist_image", "image", "wrist_image"):
        if key in out and out[key] is not None:
            out[key] = np.asarray(mirror_images_jnp(jnp.asarray(out[key])))

    image_dict = out.get("image")
    if isinstance(image_dict, dict):
        image_dict = dict(image_dict)
        for cam_key in ("base_0_rgb", "left_wrist_0_rgb"):
            if cam_key in image_dict and image_dict[cam_key] is not None:
                image_dict[cam_key] = np.asarray(
                    mirror_images_jnp(jnp.asarray(image_dict[cam_key]))
                )
        out["image"] = image_dict

    # NOTE: removed out["actions"] from mirroring, probably it is wrong.
    # if "actions" in out and out["actions"] is not None:
    #    out["actions"] = np.asarray(mirror_action_chunks_jnp(out["actions"]))

    return out


@dataclasses.dataclass(frozen=True)
class Libero90MirrorTransform(transforms.DataTransformFn):
    """Data transform that mirrors LIBERO-90 images/actions."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return _mirror_libero90_example(data)


# Backward compatibility: PairDataset replaces AugmentedDataset and CotrainingDataset
AugmentedDataset = PairDataset  # paired=True
CotrainingDataset = PairDataset  # paired=False (pass paired=False when constructing)

__all__ = [
    "PairDataset",
    "AugmentedDataset",
    "CotrainingDataset",
    "FilteredDataset",
    "MakePseudoLabelTransform",
    "LIBERO_90_TASK_IDS_MAPPING",
    "Libero90MirrorTransform",
    "convert_to_lerobot_dataset",
    "create_torch_data_loader",
    "extract_prompt_from_filename",
    "make_pseudo_label_inference_fn",
    "mirror_action_chunks_jnp",
    "mirror_images_jnp",
    "override_create_torch_dataset",
    "prepare_task_dataset",
    "augment_dataset_with_pseudo_labels",
]
