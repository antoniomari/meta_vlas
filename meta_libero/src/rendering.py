"""Rendering utilities used by TTT and debugging."""

import logging

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import openpi.models.model as _model
from openpi.policies import policy as _policy
from openpi_client import image_tools


def decode_tokenized_prompt(
    tokenized_prompt: np.ndarray | jnp.ndarray | None,
    tokenized_prompt_mask: np.ndarray | jnp.ndarray | None = None,
    max_token_len: int = 200,
    fallback_text: str = "Task",
) -> str:
    """Decode tokenized prompt to human-readable text."""
    if tokenized_prompt is None:
        return fallback_text

    try:
        from openpi.models import tokenizer as _tokenizer

        tok = _tokenizer.PaligemmaTokenizer(max_len=max_token_len)
        tokens = np.array(tokenized_prompt)
        if len(tokens.shape) > 1:
            tokens = tokens.flatten()

        if tokenized_prompt_mask is not None:
            mask = np.array(tokenized_prompt_mask)
            if len(mask.shape) > 1:
                mask = mask.flatten()
            valid_tokens = tokens[mask].tolist()
        else:
            valid_tokens = [int(t) for t in tokens if int(t) != 0]

        if not valid_tokens:
            return fallback_text

        decoded_text = tok._tokenizer.decode(valid_tokens).strip()
        if decoded_text.startswith("<bos>"):
            decoded_text = decoded_text[5:].strip()
        return decoded_text
    except Exception as exc:  # pragma: no cover - best effort logging
        logging.warning("Failed to decode tokenized prompt: %s", exc)
        return fallback_text


def _observation_batch_size(observation: _model.Observation) -> int:
    """Return batch size from observation (1 if single observation)."""
    img = observation.images.get("base_0_rgb")
    if img is None:
        img = observation.images.get("left_wrist_0_rgb")
    if img is None:
        return 1
    arr = np.array(img)
    if arr.ndim >= 4 and arr.shape[0] > 1:
        return int(arr.shape[0])
    return 1


def plot_observation_with_decoded_prompt(
    observation: _model.Observation,
    similarity_score: float | np.ndarray | jnp.ndarray | None = None,
    current_task_description: str | None = None,
    max_token_len: int = 200,
    plot_title_prefix: str = "Observation",
) -> None:
    """Plot observation(s) with decoded task description."""
    batch_size = _observation_batch_size(observation)
    max_plot_rows = 6
    plot_batch_size = min(batch_size, max_plot_rows)
    scores = (
        np.atleast_1d(similarity_score)
        if similarity_score is not None
        else np.full(batch_size, np.nan)
    )
    if scores.size == 1 and batch_size > 1:
        scores = np.broadcast_to(scores, (batch_size,))
    scores = np.asarray(scores).reshape(-1)[:plot_batch_size]

    images1 = observation.images.get("base_0_rgb")
    images2 = observation.images.get("left_wrist_0_rgb")
    assert images1 is not None and images2 is not None

    fig, axes = plt.subplots(
        plot_batch_size, 2, figsize=(8, 4 * plot_batch_size), squeeze=False
    )

    for i in range(plot_batch_size):
        token_prompt = observation.tokenized_prompt
        token_mask = observation.tokenized_prompt_mask
        assert token_prompt is not None
        tp = np.array(token_prompt)
        tm = np.array(token_mask) if token_mask is not None else None
        token_prompt_i = tp[i]
        token_mask_i = tm[i] if tm is not None else None

        decoded_task_description = decode_tokenized_prompt(
            tokenized_prompt=token_prompt_i,
            tokenized_prompt_mask=token_mask_i,
            max_token_len=max_token_len,
            fallback_text=current_task_description or "Task",
        )

        title_parts = [plot_title_prefix]
        if i < len(scores) and not np.isnan(scores[i]):
            title_parts.append(f"(similarity: {scores[i]:.4f})")
        if plot_batch_size > 1:
            title_parts.append(f"[{i}]")
        title_parts.append(f": {decoded_task_description}")
        row_title = " ".join(title_parts)

        # Images are in [-1, 1] for observation visualizations.
        img1 = (images1[i] + 1.0) / 2.0
        img2 = (images2[i] + 1.0) / 2.0

        for col, img in enumerate((img1, img2)):
            ax = axes[i, col]
            ax.imshow(img)
            ax.set_title(decoded_task_description, fontsize=10)
            ax.axis("off")

        axes[i, 0].set_ylabel(row_title, fontsize=9, labelpad=4)

    suptitle = plot_title_prefix
    if batch_size > plot_batch_size:
        suptitle = (
            f"{plot_title_prefix} (showing first {plot_batch_size}/{batch_size})"
        )
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def make_observation_from_simulator(
    policy: _policy.Policy, curr_obs_dict: dict
) -> _model.Observation:
    """Create an Observation object from current simulator observation dict."""
    inputs = jax.tree.map(lambda x: x, curr_obs_dict)
    inputs = policy._input_transform(inputs)
    if not policy._is_pytorch_model:
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    else:
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(policy._pytorch_device)[
                None, ...
            ],
            inputs,
        )
    return _model.Observation.from_dict(inputs)


def _draw_step_on_frame(frame: np.ndarray, step: int) -> np.ndarray:
    """Draw step number on top of a video frame."""
    out = np.asarray(frame).copy()
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()
    text = f"Step: {step}"
    pad = 6
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        w, h = max(60, len(text) * 10), 28
    draw.rectangle(
        [(0, 0), (w + 2 * pad, h + 2 * pad)],
        fill=(0, 0, 0),
        outline=(255, 255, 255),
    )
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return np.array(pil)


def render_image(img: np.ndarray, resize_size: int = 224) -> np.ndarray:
    """Rotate image 180 degrees, resize, and convert to uint8."""
    img = np.ascontiguousarray(img[::-1, ::-1])
    img_resized = image_tools.resize_with_pad(img, resize_size, resize_size)
    return image_tools.convert_to_uint8(img_resized)

