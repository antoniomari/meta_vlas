"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
from pathlib import Path
import re

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "antoniomari/libero_90"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_lm_90"
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def _find_last_parquet_episode(output_path: Path) -> int:
    """Return max episode index written as parquet, or -1 if none."""
    pattern = re.compile(r"episode_(\d+)\.parquet$")
    max_episode = -1
    for parquet_path in output_path.glob("data/chunk-*/episode_*.parquet"):
        match = pattern.search(parquet_path.name)
        if match:
            max_episode = max(max_episode, int(match.group(1)))
    return max_episode


def _build_resume_dataset_writer(
    output_path: Path,
    image_writer_threads: int,
    image_writer_processes: int,
) -> tuple[LeRobotDataset, int]:
    """
    Build a LeRobotDataset writer that resumes from existing local metadata
    without reloading all previous parquet episodes into memory.
    """
    meta = LeRobotDatasetMetadata(repo_id=REPO_NAME, root=output_path)
    next_episode_index = meta.total_episodes
    last_fully_written = next_episode_index - 1
    last_parquet_written = _find_last_parquet_episode(output_path)

    print(f"[resume] last fully written episode (metadata): {last_fully_written}")
    print(f"[resume] last parquet episode found on disk: {last_parquet_written}")
    print(f"[resume] starting from episode index: {next_episode_index}")

    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.meta = meta
    dataset.repo_id = meta.repo_id
    dataset.root = meta.root
    dataset.revision = None
    dataset.tolerance_s = 1e-4
    dataset.image_writer = None
    if image_writer_processes or image_writer_threads:
        dataset.start_image_writer(image_writer_processes, image_writer_threads)
    dataset.episode_buffer = dataset.create_episode_buffer(episode_index=next_episode_index)
    dataset.episodes = None
    dataset.hf_dataset = dataset.create_hf_dataset()
    dataset.image_transforms = None
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    dataset.episode_data_index = None
    dataset.video_backend = None
    return dataset, next_episode_index


def main(
    data_dir: str,
    *,
    push_to_hub: bool = False,
    resume: bool = True,
    image_writer_threads: int = 2,
    image_writer_processes: int = 1,
):
    output_path = HF_LEROBOT_HOME / REPO_NAME

    if output_path.exists() and not resume:
        shutil.rmtree(output_path)

    assert output_path.exists(), f"Output path {output_path} does not exist"
    assert resume, "Not resume?? That's not right"
    if output_path.exists() and resume:
        dataset, start_episode_index = _build_resume_dataset_writer(
            output_path=output_path,
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
        )
    else:
        start_episode_index = 0
        # Create LeRobot dataset, define features to store
        # OpenPi assumes that proprio is stored in `state` and actions in `action`
        # LeRobot assumes that dtype of image data is `image`
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            robot_type="panda",
            fps=10,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (224, 224, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (224, 224, 3),
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
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
        )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    global_episode_index = 0
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            if global_episode_index < start_episode_index:
                global_episode_index += 1
                continue

            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()
            global_episode_index += 1

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
