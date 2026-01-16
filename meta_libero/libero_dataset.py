## Training Function
import contextlib
import copy
import dataclasses
import functools
from typing import Any, Iterator, SupportsIndex, Tuple, List, Dict
import os
import logging
import etils.epath as epath
from pathlib import Path
# Evaluate pretrained model on first task of libero_90
import collections
import math
import pathlib
import imageio
import sys
import random
import functools
import h5py

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
    T_co
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

from torch.utils.data import Dataset

class FilteredDataset(Dataset):
    """Wraps a dataset and filters samples by task_index."""

    def __init__(self, dataset: Dataset, task_index: int):
        self.dataset = dataset
        self.task_index = task_index

        # Build index mapping: only samples with matching task_index
        self.indices = []
        print(f"Filtering dataset for task_index={task_index}...")

        # Fast path: Access metadata directly from LeRobot dataset
        if hasattr(dataset, 'hf_dataset') and 'task_index' in dataset.hf_dataset.column_names:
            # Direct access to HuggingFace dataset column (much faster!)
            task_indices = dataset.hf_dataset['task_index']
            self.indices = [i for i, ti in enumerate(task_indices) if ti == task_index]
            print(f"Filtered dataset (fast): {len(self.indices)} / {len(dataset)} samples")
        else:
            # Fallback: iterate through samples (slow)
            print("Warning: Falling back to slow filtering (no direct metadata access)")
            for i in range(len(dataset)):
                sample = dataset[i]
                if sample.get('task_index') == task_index:
                    self.indices.append(i)
            print(f"Filtered dataset: {len(self.indices)} / {len(dataset)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


@contextlib.contextmanager
def override_create_torch_dataset(repo_id: str, task_index: int | None = None, load_in_memory: bool = False):
    """Context manager to temporarily override create_torch_dataset in data_loader module.

    Args:
        repo_id: The LeRobot dataset repo_id
        task_index: If provided, filter to only this task_index
        load_in_memory: If True, load entire dataset into RAM (faster but uses more memory)
    """
    original = _data_loader.create_torch_dataset

    def create_dataset(
        data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
    ) -> Dataset:

        repo_id = data_config.repo_id
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
            },
        )

        # Load entire dataset into memory if requested
        if load_in_memory:
            print(f"Loading dataset '{repo_id}' into memory...")
            # Access the underlying HuggingFace dataset and load it
            if hasattr(dataset, 'hf_dataset'):
                dataset.hf_dataset = dataset.hf_dataset.with_format(None)  # Remove any format
                dataset.hf_dataset.set_format(type=None)  # Ensure no lazy loading
                # Force load all data into memory by accessing all rows
                _ = dataset.hf_dataset[:]  # This loads everything
                print(f"Dataset loaded into memory: {len(dataset)} samples")

        # Filter by task_index if specified
        if task_index is not None:
            dataset = FilteredDataset(dataset, task_index)

        if data_config.prompt_from_task:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

        return dataset

    _data_loader.create_torch_dataset = create_dataset

    try:
        yield
    finally:
        _data_loader.create_torch_dataset = original


def extract_prompt_from_filename(name: str) -> str:
    """
    Extracts the prompt (task description) from a LIBERO HDF5 filename.
    Example:
        filename = 'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo.hdf5'
        returns 'close the top drawer of the cabinet and put the black bowl on top of it'
    """
    import re
    assert name.endswith('_demo.hdf5')
    # Remove 'KITCHEN_SCENE' followed by digits and an underscore at the start
    prompt = re.sub(r'^[A-Z_]+_SCENE\d+_', '', name)
    prompt = re.sub(r'_demo\.hdf5$', '', prompt).replace('_', ' ')
    return prompt


# Convert LIBERO data to model format
def prepare_task_dataset(task_suite_name: str = "libero_90", task_id: int = 0) -> List[List[Dict]]:

    DATASET_DIR = "/cluster/scratch/anmari/libero_datasets"
    IMAGE_RESOLUTION = (256, 256)
    libero_path = Path(DATASET_DIR) / task_suite_name
    hdf5_files = sorted(list(libero_path.glob("*.hdf5")))

    if not hdf5_files:
        raise ValueError("No HDF5 files found in libero_90!")

    if task_id >= len(hdf5_files):
        raise ValueError(f"Task ID {task_id} out of range. Only {len(hdf5_files)} tasks available.")

    task_file = hdf5_files[task_id]
    print(f"\nLoading task {task_id}: {task_file.name}")
    # Extract prompt from task file name
    prompt = extract_prompt_from_filename(task_file.name)

    # Load all episodes from the task
    episodes = []
    with h5py.File(task_file, "r") as f:
        demo_keys = [key for key in f["data"].keys() if key.startswith("demo_")]
        for demo_key in demo_keys:
            demo_data = f["data"][demo_key]
            episode = {
                'observations': {},
                'actions': np.array(demo_data['actions']),
                'prompt': prompt,
                'state': np.concatenate([np.array(demo_data['obs']['ee_pos']), np.array(demo_data['obs']['ee_ori']), np.array(demo_data['obs']['gripper_states'])], axis=1),
            }
            # Load all camera views
            for camera_view in ['agentview_rgb', 'eye_in_hand_rgb']:
                if camera_view in demo_data['obs']:
                    episode['observations'][camera_view] = np.array(demo_data['obs'][camera_view])
            episodes.append(episode)

    print(f"Loaded {len(episodes)} episodes from task {task_id}")

    # Iterate on episodes, preprocess for model format
    dataset = []
    for episode in episodes:
        episode_data = []
        # Get the length from actions (should match observations)
        episode_len = len(episode['actions'])

        # Process each timestep
        for t in range(episode_len):


            image = episode['observations']['agentview_rgb'][t]
            image = image[::-1, ::-1]
            image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(image, IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1])
            )
            wrist_image = episode['observations']['eye_in_hand_rgb'][t]
            wrist_image = wrist_image[::-1, ::-1]
            wrist_image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_image, IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1])
            )

            episode_data.append({
                'image': image,
                'wrist_image': wrist_image,
                'state': episode['state'][t],
                'actions': episode['actions'][t],
                'prompt': episode['prompt'],
            })

        dataset.append(episode_data)

    return dataset


def convert_to_lerobot_dataset(dataset: List[List[Dict]], repo_id: str = "example") -> LeRobotDataset:
    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    action_horizon = 10
    fps = 10

    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for episode in dataset:
        for sample in episode:
            lerobot_dataset.add_frame(
                {
                    "image": sample['image'],
                    "wrist_image": sample['wrist_image'],
                    "state": sample['state'].astype(np.float32),
                    "actions": sample['actions'].astype(np.float32),
                    "task": sample['prompt'],
                }
            )
        lerobot_dataset.save_episode()

    return lerobot_dataset



@contextlib.contextmanager
def override_create_torch_dataset_old(repo_id: str):
    """Context manager to temporarily override create_torch_dataset in data_loader module.

    Usage:
        with override_create_torch_dataset():
            # Now all calls to data_loader.create_torch_dataset will use your override
            loader = _data_loader.create_torch_data_loader(...)

    Or with custom function:
        with override_create_torch_dataset(my_custom_create_torch_dataset):
            loader = _data_loader.create_torch_data_loader(...)
    """
    # Save original
    original = _data_loader.create_torch_dataset


    def create_dataset(
        data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
    ) -> Dataset:

        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,  # Use the repo_id parameter consistently
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
            },
        )

        if data_config.prompt_from_task:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

        return dataset

    _data_loader.create_torch_dataset = create_dataset

    try:
        yield
    finally:
        # Restore original
        _data_loader.create_torch_dataset = original


def create_data_loader(config, repo_id: str, task_index: int | None = None, load_in_memory: bool = False):

    if repo_id is None:
        data_loader = _data_loader.create_data_loader(
                config,
                sharding=None,
                shuffle=True,
            )
    else:
        with override_create_torch_dataset(repo_id, task_index=task_index, load_in_memory=load_in_memory):
            data_loader = _data_loader.create_data_loader(
                    config,
                    sharding=None,
                    shuffle=True,
                )
    return data_loader


# Config used for pi05-libero
"""
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
"""

