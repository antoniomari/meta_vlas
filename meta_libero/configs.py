"""Centralised hyperparameter / settings dataclasses for meta_libero.

Each dataclass is the **single source of truth** for its defaults.
Scripts build instances from argparse; notebooks instantiate directly;
or load a full experiment from a YAML file.

Usage examples
--------------
# In a notebook:
    from configs import TTTConfig, EvalConfig, ExperimentConfig

    ttt_cfg  = TTTConfig(learning_rate=1e-4, k=20)
    eval_cfg = EvalConfig(task_id=3, num_trials=50)

# From a YAML experiment file:
    from configs import load_experiment

    exp = load_experiment("experiments/example_ttt.yaml")
    exp.ttt.learning_rate   # -> 1e-4
    exp.eval.num_trials     # -> 50

# From a script with --config support:
    exp = build_experiment_from_cli()   # handles --config + CLI overrides
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Model / checkpoint selection
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Flags that control *which* model weights and dataset to use."""

    use_base_model: bool = False
    """Use base pi0.5 weights instead of the Libero-pretrained checkpoint."""

    use_lora: bool = False
    """Use LoRA-adapted weights."""

    action_expert_only: bool = False
    """Only fine-tune the action-expert sub-network."""

    libero_90_dataset: bool = False
    """Use the libero_90 dataset instead of libero_10."""

    @property
    def dataset_to_use(self) -> str:
        """Convenience: returns ``'libero_90'`` or ``'libero_10'``."""
        return "libero_90" if self.libero_90_dataset else "libero_10"


# ---------------------------------------------------------------------------
# TTT (test-time training) hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TTTConfig:
    """Hyperparameters that control test-time training during rollout."""

    learning_rate: float = 2.5e-5
    """Learning rate for TTT gradient steps."""

    ttt_num_steps: int = 10
    """Number of gradient steps per TTT update."""

    ttt_frequency: int = 50
    """Perform a TTT update every *frequency* environment steps."""

    k: int = 50
    """Number of nearest neighbours to retrieve for each TTT update."""

    ttt_max_step: int = 1000
    """No further TTT updates after this environment step."""

    use_modalities: Optional[List[str]] = None
    """Modalities used for nearest-neighbour retrieval (default: all available)."""

    repeat_batch: int = 1
    """How many times to repeat retrieved samples in the TTT batch."""

    reset_policy: bool = True
    """Whether to reset the policy weights before each evaluation episode."""

    random_neighbors: bool = False
    """Retrieve random samples instead of nearest neighbours (ablation)."""

    cfg_weight: float = 1.0
    """Classifier-free guidance weight applied during TTT inference."""

    noise_ttt: bool = False
    """Use Gaussian-noise perturbation instead of gradient-based TTT."""


# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Settings that control *how* an evaluation rollout is run."""

    num_trials: int = 10
    """Number of evaluation episodes per task."""

    task_suite_name: str = "libero_90"
    """Name of the LIBERO task suite (e.g. ``libero_10``, ``libero_90``)."""

    task_id: int = 0
    """Task index within the suite."""

    num_steps_wait: int = 10
    """Idle steps at the start of each episode (environment stabilisation)."""

    save_video: bool = False
    """Whether to record rollout videos."""

    video_out_path: str = "data/libero/videos"
    """Directory where rollout videos are written."""

    seed: int = 0
    """Random seed for reproducibility."""

    plot_observations: bool = False
    """If ``True``, render per-step observation plots (slow, for debugging)."""

    use_test_task: bool = False
    """Use the held-out test task definition instead of the training one."""


# ---------------------------------------------------------------------------
# Fine-tuning / offline training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class FinetuneConfig:
    """Hyperparameters for offline fine-tuning (``train_model_on_fly``)."""

    learning_rate: float = 2.5e-5
    """Optimizer learning rate."""

    num_steps: int = 1000
    """Total number of gradient steps to train for."""

    eval_interval: int = 100
    """Evaluate every *eval_interval* gradient steps."""

    batch_size: int = 64
    """Training batch size."""

    warmup_steps: int = 100
    """Linear warmup steps for the learning-rate schedule."""

    weight_decay: float = 0.0
    """AdamW weight-decay coefficient."""

    log_interval: int = 100
    """Print training metrics every *log_interval* steps."""

    seed: int = 42
    """Random seed for reproducibility."""

    donate_buffers: bool = True
    """Pass input buffers to XLA by donation (faster, but the caller
    must not reuse them).  Set ``False`` when you need the model
    immediately after training (e.g. TTT)."""

    skip_first_eval: bool = False
    """Skip the evaluation run at step 0 (before any training)."""


# ---------------------------------------------------------------------------
# Noise-perturbation config (for run_evaluation_noise)
# ---------------------------------------------------------------------------

@dataclass
class NoiseConfig:
    """Settings for the Gaussian-noise perturbation baseline."""

    sigma: float = 0.01
    """Standard deviation of the Gaussian noise added to parameters."""

    frequency: int = 50
    """Apply noise every *frequency* environment steps."""

    max_step: int = 150
    """No further perturbations after this environment step."""


# ---------------------------------------------------------------------------
# Experiment-level bundle
# ---------------------------------------------------------------------------

# Maps YAML section names → dataclass types.
_SECTION_MAP: Dict[str, type] = {
    "model": ModelConfig,
    "ttt": TTTConfig,
    "eval": EvalConfig,
    "finetune": FinetuneConfig,
    "noise": NoiseConfig,
}


@dataclass
class ExperimentConfig:
    """Bundles every config needed for one experiment.

    Sections that are absent from the YAML file get their dataclass defaults.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    ttt: TTTConfig = field(default_factory=TTTConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    # Optional free-form metadata (experiment name, notes, tags, …).
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# YAML load / save
# ---------------------------------------------------------------------------

def load_experiment(path: Union[str, Path]) -> ExperimentConfig:
    """Load an ``ExperimentConfig`` from a YAML file.

    Each top-level key in the YAML (``model``, ``ttt``, ``eval``,
    ``finetune``, ``noise``) is mapped to the corresponding dataclass.
    Missing sections fall back to dataclass defaults.  An optional
    ``meta`` section is kept as a plain dict.

    Parameters
    ----------
    path:
        Path to a ``.yaml`` file.

    Returns
    -------
    ExperimentConfig
    """
    path = Path(path)
    with path.open("r") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    kwargs: Dict[str, Any] = {}
    for section_name, cls in _SECTION_MAP.items():
        section_data = raw.get(section_name, {})
        if section_data:
            kwargs[section_name] = cls(**section_data)
        # else: will use field default_factory

    if "meta" in raw:
        kwargs["meta"] = raw["meta"]

    return ExperimentConfig(**kwargs)


def save_experiment(config: ExperimentConfig, path: Union[str, Path]) -> Path:
    """Serialise an ``ExperimentConfig`` to a YAML file.

    Parameters
    ----------
    config:
        The experiment config to save.
    path:
        Destination ``.yaml`` file (parent dirs are created automatically).

    Returns
    -------
    Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {}
    for section_name in _SECTION_MAP:
        section_obj = getattr(config, section_name)
        data[section_name] = dataclasses.asdict(section_obj)

    if config.meta:
        data["meta"] = config.meta

    with path.open("w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    return path


# ---------------------------------------------------------------------------
# CLI helpers — build an ExperimentConfig from argparse + optional YAML
# ---------------------------------------------------------------------------

def _override_dataclass(base: Any, overrides: Dict[str, Any]) -> Any:
    """Return a copy of *base* with non-None values from *overrides* applied.

    Only keys that exist as dataclass fields are considered.
    """
    valid = {f.name for f in dataclasses.fields(base)}
    changes = {k: v for k, v in overrides.items() if k in valid and v is not None}
    if changes:
        return dataclasses.replace(base, **changes)
    return base


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register ``--config`` and model-selection flags on *parser*.

    These arguments are shared between TTT and fine-tuning scripts.
    Call this once at the top of each script's ``main()`` before
    ``parse_args()``.
    """
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML experiment config (overrides defaults; "
             "explicit CLI flags override YAML values)",
    )
    # ModelConfig flags
    parser.add_argument("--use-base-model", action="store_true", default=None,
                        help="Use base pi0.5 weights instead of Libero-pretrained")
    parser.add_argument("--use-lora", action="store_true", default=None,
                        help="Use LoRA weights")
    parser.add_argument("--action-expert-only", action="store_true", default=None,
                        help="Only finetune action expert")
    parser.add_argument("--libero_90_dataset", action="store_true", default=None,
                        help="Use libero_90 dataset")
    # EvalConfig flags shared across scripts
    parser.add_argument("--task_suite_name", type=str, default=None,
                        help="Task suite name (e.g. libero_10, libero_90)")
    parser.add_argument("--task_id", type=int, default=None,
                        help="Task ID within the suite")
    parser.add_argument("--num_trials", type=int, default=None,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--save-video", action="store_true", default=None,
                        help="Save rollout videos")


def build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build an ``ExperimentConfig`` from parsed CLI args.

    If ``args.config`` is set, loads the YAML first, then applies any
    explicitly-provided CLI flags on top (non-None values override YAML).
    If ``args.config`` is not set, starts from dataclass defaults and
    applies CLI flags.

    Parameters
    ----------
    args:
        Parsed ``argparse.Namespace``.  Expected to contain at least the
        flags registered by :func:`add_common_args`.

    Returns
    -------
    ExperimentConfig
    """
    # Start from YAML or defaults
    if args.config:
        exp = load_experiment(args.config)
    else:
        exp = ExperimentConfig()

    # -- ModelConfig overrides -----------------------------------------------
    model_overrides: Dict[str, Any] = dict(
        use_base_model=getattr(args, "use_base_model", None),
        use_lora=getattr(args, "use_lora", None),
        action_expert_only=getattr(args, "action_expert_only", None),
        libero_90_dataset=getattr(args, "libero_90_dataset", None),
    )
    exp = dataclasses.replace(exp, model=_override_dataclass(exp.model, model_overrides))

    # -- EvalConfig overrides ------------------------------------------------
    eval_overrides: Dict[str, Any] = dict(
        task_suite_name=getattr(args, "task_suite_name", None),
        task_id=getattr(args, "task_id", None),
        num_trials=getattr(args, "num_trials", None),
        seed=getattr(args, "seed", None),
        save_video=getattr(args, "save_video", None),
    )
    exp = dataclasses.replace(exp, eval=_override_dataclass(exp.eval, eval_overrides))

    # -- TTTConfig overrides -------------------------------------------------
    ttt_overrides: Dict[str, Any] = dict(
        learning_rate=getattr(args, "lr", None),
        ttt_num_steps=getattr(args, "ttt_num_steps", None),
        ttt_frequency=getattr(args, "ttt_frequency", None),
        k=getattr(args, "ttt_k", None),
        ttt_max_step=getattr(args, "max_ttt_step", None),
        noise_ttt=getattr(args, "noise_ttt", None),
    )
    # --no-reset-policy  →  reset_policy = False
    no_reset = getattr(args, "no_reset_policy", None)
    if no_reset is not None:
        ttt_overrides["reset_policy"] = not no_reset
    exp = dataclasses.replace(exp, ttt=_override_dataclass(exp.ttt, ttt_overrides))

    # -- FinetuneConfig overrides --------------------------------------------
    ft_overrides: Dict[str, Any] = dict(
        learning_rate=getattr(args, "lr", None),
        num_steps=getattr(args, "total_steps", None),
        eval_interval=getattr(args, "eval_interval", None),
        batch_size=getattr(args, "batch_size", None),
        warmup_steps=getattr(args, "warmup_steps", None),
        seed=getattr(args, "seed", None),
        skip_first_eval=getattr(args, "skip_first_eval", None),
    )
    exp = dataclasses.replace(exp, finetune=_override_dataclass(exp.finetune, ft_overrides))

    # -- NoiseConfig overrides -----------------------------------------------
    noise_overrides: Dict[str, Any] = dict(
        sigma=getattr(args, "noise_sigma", None),
        frequency=getattr(args, "noise_frequency", None),
        max_step=getattr(args, "noise_max_step", None),
    )
    exp = dataclasses.replace(exp, noise=_override_dataclass(exp.noise, noise_overrides))

    return exp
