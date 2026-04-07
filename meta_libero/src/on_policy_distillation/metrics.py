"""Trainable parameter change metrics (student BC phases)."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


def trainable_param_delta_l2_norm(model_before, model_after, train_config) -> float:
    """L2 norm of (θ_after − θ_before) over trainable parameters (same filter as BC training)."""
    st0 = nnx.state(model_before)
    st1 = nnx.state(model_after)
    tf = train_config.trainable_filter
    if tf is not None:
        p0 = st0.filter(tf)
        p1 = st1.filter(tf)
    else:
        p0, p1 = st0, st1

    def to_float32_flat(x: Any) -> Any:
        v = x.value if hasattr(x, "value") else x
        return jnp.ravel(jnp.asarray(v, dtype=jnp.float32))

    leaves0 = jax.tree.map(to_float32_flat, p0)
    leaves1 = jax.tree.map(to_float32_flat, p1)
    diffs = jax.tree.map(lambda a, b: a - b, leaves0, leaves1)
    flat = jnp.concatenate([d for d in jax.tree_util.tree_leaves(diffs)])
    return float(jax.device_get(jnp.linalg.norm(flat)))
