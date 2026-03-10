"""Canonical dataset utilities for meta_libero."""

import collections
import contextlib
import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import jax.numpy as jnp
import numpy as np
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # type: ignore
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
import openpi.models.model as _model  # type: ignore
from openpi.training import config as _config  # type: ignore
import openpi.training.data_loader as _data_loader  # type: ignore
from openpi.training.data_loader import Dataset, TransformedDataset  # type: ignore
import openpi.transforms as transforms  # type: ignore
from openpi_client import image_tools  # type: ignore


_TASK_INDICES_CACHE: dict[str, dict[int, list[int]]] = {}

LIBERO_90_TASK_IDS_MAPPING = {
    0: 55,
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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


@contextlib.contextmanager
def override_create_torch_dataset(
    repo_id: Optional[str] = None,
    task_id: int | None = None,
    load_in_memory: bool = False,
    mirror_data: bool = True,
):
    """Temporarily override OpenPI create_torch_dataset with LIBERO-aware one."""
    original = _data_loader.create_torch_dataset

    def create_dataset(
        data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
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

        if task_id is not None:
            if repo_id == "antoniomari/libero_90":
                if task_id not in LIBERO_90_TASK_IDS_MAPPING:
                    raise AssertionError(f"Task ID {task_id} not found in LIBERO_90_TASK_IDS_MAPPING")
                task_id_dataset = LIBERO_90_TASK_IDS_MAPPING[task_id]
            else:
                raise NotImplementedError(f"Only supported for antoniomari/libero_90, got {repo_id}")
            dataset = FilteredDataset(dataset, task_id_dataset, repo_id=repo)

        if data_config.prompt_from_task:
            dataset = TransformedDataset(dataset, [transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

        # Apply LIBERO-90 mirroring directly in the loader override when enabled.
        if mirror_data and repo == "antoniomari/libero_90":
            dataset = TransformedDataset(dataset, [Libero90MirrorTransform()])

        return dataset

    _data_loader.create_torch_dataset = create_dataset
    try:
        yield
    finally:
        _data_loader.create_torch_dataset = original


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
    print(arr.shape)
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


__all__ = [
    "FilteredDataset",
    "LIBERO_90_TASK_IDS_MAPPING",
    "Libero90MirrorTransform",
    "convert_to_lerobot_dataset",
    "extract_prompt_from_filename",
    "mirror_action_chunks_jnp",
    "mirror_images_jnp",
    "override_create_torch_dataset",
    "prepare_task_dataset",
]
