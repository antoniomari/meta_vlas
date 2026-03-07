import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BatchableLiberoInputs(transforms.DataTransformFn):
    """Libero input transform with batch-shaped image masks."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        right_wrist = np.zeros_like(base_image)

        batch_shape = base_image.shape[:-3]
        true_mask = np.ones(batch_shape, dtype=bool)
        false_mask = np.zeros(batch_shape, dtype=bool)

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": true_mask.copy(),
                "left_wrist_0_rgb": true_mask.copy(),
                "right_wrist_0_rgb": true_mask.copy()
                if self.model_type == _model.ModelType.PI0_FAST
                else false_mask.copy(),
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "task_index" in data:
            inputs["task_index"] = data["task_index"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BatchableLiberoOutputs(transforms.DataTransformFn):
    """Libero output transform matching OpenPI LiberoOutputs behavior."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}

