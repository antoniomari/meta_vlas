"""BC buffer metadata: alignment / teacher-variance filters and strip."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_BC_META_KEYS = frozenset(
    {"alignment_ratio", "replan_env_step", "teacher_chunk_var_mean"}
)


def example_alignment_ratio(ex: dict) -> float:
    v = ex.get("alignment_ratio")
    if v is None:
        return float("nan")
    return float(np.asarray(v, dtype=np.float64).reshape(-1)[0])


def example_env_step(ex: dict) -> int | None:
    """Env step at replan for this distillation row (see ttt distillation metadata)."""
    v = ex.get("replan_env_step")
    if v is None:
        return None
    return int(np.asarray(v, dtype=np.int64).reshape(-1)[0])


def example_teacher_chunk_var_mean(ex: dict) -> float:
    """Mean_{h,d} Var_{teacher samples}(chunk[h,d]); missing key -> 0 (legacy rows)."""
    v = ex.get("teacher_chunk_var_mean")
    if v is None:
        return 0.0
    return float(np.asarray(v, dtype=np.float64).reshape(-1)[0])


def kept_by_alignment_policy(
    ratio: float,
    *,
    alignment_ratio_threshold: float | None,
    align_min: float | None,
) -> bool:
    """Keep sample: either ratio <= threshold (augment convention) or ratio >= align_min."""
    if alignment_ratio_threshold is None and align_min is None:
        return True
    if not math.isfinite(ratio):
        return True
    if alignment_ratio_threshold is not None:
        return ratio <= alignment_ratio_threshold
    assert align_min is not None
    return ratio >= align_min


def filter_examples_by_alignment(
    examples: list[dict],
    *,
    alignment_ratio_threshold: float | None = None,
    align_min: float | None = None,
) -> tuple[list[dict], int, int]:
    """Return (kept examples, n_kept, n_dropped)."""
    if alignment_ratio_threshold is not None and align_min is not None:
        raise ValueError("only one of alignment_ratio_threshold and align_min may be set")
    if alignment_ratio_threshold is None and align_min is None:
        return list(examples), len(examples), 0
    kept: list[dict] = []
    dropped = 0
    for ex in examples:
        if kept_by_alignment_policy(
            example_alignment_ratio(ex),
            alignment_ratio_threshold=alignment_ratio_threshold,
            align_min=align_min,
        ):
            kept.append(ex)
        else:
            dropped += 1
    return kept, len(kept), dropped


def filter_examples_by_max_teacher_variance(
    examples: list[dict],
    *,
    max_teacher_variance: float | None,
) -> tuple[list[dict], int, int]:
    """Drop samples with teacher chunk variance (mean over H,D of var across teacher samples) > cap."""
    if max_teacher_variance is None:
        return list(examples), len(examples), 0
    kept: list[dict] = []
    dropped = 0
    cap = float(max_teacher_variance)
    for ex in examples:
        if example_teacher_chunk_var_mean(ex) <= cap:
            kept.append(ex)
        else:
            dropped += 1
    return kept, len(kept), dropped


def strip_bc_metadata(ex: dict) -> dict:
    return {k: v for k, v in ex.items() if k not in _BC_META_KEYS}


# Backward-compatible aliases for code that used private names on the script module.
_filter_examples_by_alignment = filter_examples_by_alignment
_strip_bc_metadata = strip_bc_metadata
_example_alignment_ratio = example_alignment_ratio
_kept_by_alignment_policy = kept_by_alignment_policy
