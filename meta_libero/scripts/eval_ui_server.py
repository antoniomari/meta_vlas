# Interactive Evaluation Web UI for LIBERO-90
#
# Prerequisites:
#   pip install fastapi uvicorn
#
# Run:
#   cd meta_vlas && python meta_libero/scripts/eval_ui_server.py [--host 0.0.0.0] [--port 8765]
#
# Open in any browser: http://localhost:8765
#
# Reuses components from ttt_evaluation.py and src/ttt.py.

import sys
import logging
import warnings
import base64
import collections
import threading
import dataclasses

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*shape requires ndarray or scalar arguments.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax.core.scope")

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT, _REPO_ROOT / "src", _REPO_ROOT / "meta_libero", _REPO_ROOT / "third_party" / "libero"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("HF_LEROBOT_HOME", str(Path(os.environ["HF_HOME"]) / "lerobot"))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import numpy as np
import jax
import jax.numpy as jnp

from openpi_client import image_tools
from libero.libero import benchmark

from meta_libero.src.utils import _get_libero_env, _quat2axisangle, compute_alignment_ratio
from meta_libero.src.ttt import (
    load_pi05_libero_model,
    create_policy,
    _infer_action_chunk_samples,
    _build_curr_obs_dict,
    _preprocess_images,
    _max_steps_for_task_suite,
    train_model_on_fly,
    copy_model,
    merge_model_parameters,
)
from meta_libero.src.dataset import (
    override_create_torch_dataset,
    LIBERO_90_TASK_IDS_MAPPING,
    LIBERO_90_TASK_IDS_PROMPTS,
    make_pseudo_label_inference_fn,
)
from meta_libero.src.rendering import make_observation_from_simulator
from openpi.training import data_loader as _data_loader

# Lazy imports for nn_fetcher (heavy)
_nn_fetcher = None
_neighbors_dataset = None

# ---------------------------------------------------------------------------
# Module state (env, policy, etc.)
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = os.getenv(
    "OPENPI_CHECKPOINT_DIR",
    str(Path.home() / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero"),
)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
RESIZE_SIZE = 224
REPLAN_STEPS = 5
NUM_STEPS_WAIT = 10
TASK_SUITE_NAME = "libero_90"
MAX_STEPS = _max_steps_for_task_suite(TASK_SUITE_NAME)

_state = {
    "model": None,
    "config": None,
    "policy": None,
    "env": None,
    "task_suite": None,
    "task_id": None,
    "task_instruction": None,
    "initial_state": None,
    "obs": None,
    "action_plan": None,
    "t": 0,
    "done": False,
    "alignment_scores": [],
    "seed": 42,
    "finetune_status": "idle",
    "finetune_losses": [],
    "finetune_validation_losses": [],  # list of {"step": int, "loss": float}
    "finetune_error": None,
    "finetune_total_steps": 0,
    # Reference models for Task regularization (never overwritten by fine-tuning)
    # List of {id, model, policy, task_id, name}; most recent first
    "references": [],
    "reference_counter": 0,
}
MAX_REFERENCES = 5


def _ensure_model():
    """Load model and policy once."""
    if _state["model"] is None:
        print("Loading model...")
        _state["model"], _state["config"] = load_pi05_libero_model(
            use_base_model=False,
            use_lora=False,
            action_expert_only=False,
        )
        _state["policy"] = create_policy(
            _state["model"],
            _state["config"],
            CHECKPOINT_DIR,
            rng_seed=_state["seed"],
        )
        print("Model loaded")


def _ensure_task_suite():
    """Load task suite once."""
    if _state["task_suite"] is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        _state["task_suite"] = benchmark_dict[TASK_SUITE_NAME]()


def _ensure_nn_fetcher():
    """Load NN fetcher for libero_90 once."""
    global _nn_fetcher
    if _nn_fetcher is None:
        from meta_libero.nn_fetcher import NearestNeighborFetcher
        cache_dir = Path.home() / ".cache" / "libero_90_norm"
        modality_str = "_".join(sorted(["image1", "image2", "text"]))
        index_path = cache_dir / f"libero_unified_faiss_index_{modality_str}.index"
        metadata_path = cache_dir / f"libero_unified_faiss_metadata_{modality_str}.pkl"
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {cache_dir}. Run build_unified_faiss_index.py for libero_90 first."
            )
        _ensure_model()
        _nn_fetcher = NearestNeighborFetcher(
            index_path=str(index_path),
            metadata_path=str(metadata_path),
            model=_state["model"],
        )
        print("NN fetcher initialized for libero_90")
    return _nn_fetcher


def _ensure_neighbors_dataset():
    """Load full libero_90 dataset (no task filter) for neighbor lookup."""
    global _neighbors_dataset
    if _neighbors_dataset is None:
        _ensure_model()
        config = _state["config"]
        with override_create_torch_dataset(
            repo_id="antoniomari/libero_90",
            task_id=None,
            mirror_data=True,
            single_episode=False,
        ):
            dataloader = _data_loader.create_data_loader(
                config, sharding=None, shuffle=False,
            )
            _neighbors_dataset = dataloader._data_loader._data_loader.dataset
        print("Neighbors dataset loaded (full libero_90)")
    return _neighbors_dataset


def _sample_images_to_base64(sample: dict) -> tuple[str | None, str | None]:
    """Extract base and wrist images from a dataset sample and encode to base64."""
    from PIL import Image
    import io
    batch = _data_loader._collate_fn([sample])
    from openpi.models import model as _model
    obs = _model.Observation.from_dict(batch)
    base_img = obs.images.get("base_0_rgb") if hasattr(obs, "images") else None
    wrist_img = obs.images.get("left_wrist_0_rgb") if hasattr(obs, "images") else None

    def _encode(img) -> str | None:
        if img is None:
            return None
        arr = np.asarray(img)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            return None
        arr = np.ascontiguousarray(arr)
        if arr.dtype != np.uint8:
            arr = image_tools.convert_to_uint8(arr)
        arr = image_tools.resize_with_pad(arr, RESIZE_SIZE, RESIZE_SIZE)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            pil = Image.fromarray(arr)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        return None

    return _encode(base_img), _encode(wrist_img)


def _get_sample_task_index(sample: dict) -> int | None:
    """Get task_index from a dataset sample if available."""
    if "task_index" in sample:
        return int(sample["task_index"])
    return None


def _obs_to_base64(obs: dict) -> str:
    """Encode agentview image to base64 for frontend."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    img_resized = image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE)
    img_uint8 = image_tools.convert_to_uint8(img_resized)
    if img_uint8.ndim == 3 and img_uint8.shape[-1] == 3:
        from PIL import Image
        pil = Image.fromarray(img_uint8)
        import io
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    return ""


def _obs_to_base64_wrist(obs: dict) -> str:
    """Encode wrist image to base64."""
    img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img_resized = image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE)
    img_uint8 = image_tools.convert_to_uint8(img_resized)
    if img_uint8.ndim == 3 and img_uint8.shape[-1] == 3:
        from PIL import Image
        pil = Image.fromarray(img_uint8)
        import io
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    return ""


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel


class LoadEnvRequest(BaseModel):
    task_id: int
    task_instruction: str | None = None


class StepRequest(BaseModel):
    task_instruction: str | None = None
    compute_alignment: bool = True  # If False, skip alignment (removes ~1 extra inference delay every 5 steps)


class FinetuneRequest(BaseModel):
    learning_rate: float = 2.5e-5
    num_steps: int = 100
    batch_size: int = 8
    warmup_steps: int = 10
    single_episode: bool = True
    merging_eps: float | None = 1.0  # 1.0 = keep full fine-tuned model; 0.0 = keep original; 0.5 = 50/50 blend
    augment: bool = False  # augment each batch with (normal + augmented) samples
    augment_task_id: int | None = None  # None = "do nothing" + zeros; int = use task prompt
    task_regularization: bool = False  # use reference model for pseudo-labels (requires augment_task_id)
    reference_id: str | None = None  # which saved reference to use (required when task_regularization)
    add_validation_set: bool = False  # compute validation loss every 5 steps
    validation_task_id: int | None = None  # task for validation (single trajectory; required when add_validation_set)


class NeighborsRequest(BaseModel):
    filter_by_task: bool = False
    k: int = 8


app = FastAPI(title="LIBERO-90 Evaluation UI")

SCRIPT_DIR = Path(__file__).resolve().parent


@app.get("/api/tasks")
def api_tasks():
    """List all LIBERO-90 tasks with id and instruction."""
    _ensure_task_suite()
    suite = _state["task_suite"]
    tasks = []
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        tasks.append({
            "id": i,
            "instruction": task.language,
        })
    return {"tasks": tasks}


@app.post("/api/load_env")
def api_load_env(req: LoadEnvRequest):
    """Load environment for task_id and optionally set custom instruction."""
    task_id = req.task_id
    task_instruction = req.task_instruction
    _ensure_model()
    _ensure_task_suite()
    suite = _state["task_suite"]
    if task_id < 0 or task_id >= suite.n_tasks:
        raise HTTPException(status_code=400, detail=f"task_id must be 0..{suite.n_tasks - 1}")
    task = suite.get_task(task_id)
    initial_states = suite.get_task_init_states(task_id)
    initial_state = initial_states[0]

    instruction = task_instruction if task_instruction and task_instruction.strip() else task.language

    # Close old env if any
    if _state["env"] is not None:
        try:
            _state["env"].close()
        except Exception:
            pass
        _state["env"] = None

    env, _ = _get_libero_env(task, RESIZE_SIZE, _state["seed"])
    env.seed(_state["seed"])
    env.reset()
    obs = env.set_init_state(initial_state)

    _state["env"] = env
    _state["task_id"] = task_id
    _state["task_instruction"] = instruction
    _state["initial_state"] = initial_state
    _state["obs"] = obs
    _state["action_plan"] = collections.deque()
    _state["t"] = 0
    _state["done"] = False
    _state["alignment_scores"] = []

    return {
        "ok": True,
        "task_id": task_id,
        "instruction": instruction,
        "base_image": _obs_to_base64(obs),
        "wrist_image": _obs_to_base64_wrist(obs),
        "step": 0,
        "done": False,
        "alignment_scores": [],
    }


@app.post("/api/reset")
def api_reset():
    """Reset environment to initial state."""
    if _state["env"] is None or _state["initial_state"] is None:
        raise HTTPException(status_code=400, detail="Env not loaded. Load a task first.")
    env = _state["env"]
    env.seed(_state["seed"])
    env.reset()
    obs = env.set_init_state(_state["initial_state"])
    _state["obs"] = obs
    _state["action_plan"] = collections.deque()
    _state["t"] = 0
    _state["done"] = False
    _state["alignment_scores"] = []
    return {
        "ok": True,
        "base_image": _obs_to_base64(obs),
        "wrist_image": _obs_to_base64_wrist(obs),
        "step": 0,
        "done": False,
        "instruction": _state["task_instruction"],
        "alignment_scores": [],
    }


@app.post("/api/step")
def api_step(req: StepRequest = StepRequest()):
    """Run one inference + env step. Returns updated obs."""
    task_instruction = req.task_instruction
    compute_alignment = req.compute_alignment
    if _state["env"] is None or _state["obs"] is None:
        raise HTTPException(status_code=400, detail="Env not loaded. Load a task first.")
    if _state["done"]:
        return {
            "ok": True,
            "base_image": _obs_to_base64(_state["obs"]),
            "wrist_image": _obs_to_base64_wrist(_state["obs"]),
            "step": int(_state["t"]),
            "done": True,
            "reward": 1.0,
            "instruction": _state["task_instruction"],
            "alignment_ratio": None,
            "alignment_scores": [(int(s), float(r)) for s, r in _state["alignment_scores"]],
        }

    env = _state["env"]
    policy = _state["policy"]
    obs = _state["obs"]
    action_plan = _state["action_plan"]
    t = _state["t"]
    instruction = task_instruction if task_instruction and task_instruction.strip() else _state["task_instruction"]

    # Dummy steps during wait
    if t < NUM_STEPS_WAIT:
        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
        _state["obs"] = obs
        _state["t"] = t + 1
        _state["done"] = bool(done)
        return {
            "ok": True,
            "base_image": _obs_to_base64(obs),
            "wrist_image": _obs_to_base64_wrist(obs),
            "step": int(t + 1),
            "done": bool(done),
            "reward": float(reward),
            "instruction": instruction,
            "alignment_ratio": None,
            "alignment_scores": [(int(s), float(r)) for s, r in _state["alignment_scores"]],
        }

    # Get new action chunk if needed
    alignment_ratio = None
    if not action_plan:
        _state["task_instruction"] = instruction  # track instruction used for this inference
        img, wrist_img = _preprocess_images(obs, resize_size=RESIZE_SIZE)
        curr_obs_dict = _build_curr_obs_dict(obs, img, wrist_img, instruction)
        policy._rng, rng = jax.random.split(policy._rng)
        noise = jax.random.normal(rng, (1, policy._model.action_horizon, policy._model.action_dim))
        action_chunk = _infer_action_chunk_samples(policy, curr_obs_dict, noise)
        action_chunk_np = np.asarray(action_chunk)
        if action_chunk_np.ndim == 3:
            action_chunk_np = action_chunk_np[0]
        if compute_alignment:
            alignment_ratio = float(compute_alignment_ratio(policy, action_chunk, curr_obs_dict, noise=noise))
            _state["alignment_scores"].append((t, alignment_ratio))
        for i in range(min(REPLAN_STEPS, len(action_chunk_np))):
            action_plan.append(action_chunk_np[i])

    action = action_plan.popleft()
    obs, reward, done, info = env.step(action.tolist())
    _state["obs"] = obs
    _state["action_plan"] = action_plan
    _state["t"] = t + 1
    _state["done"] = bool(done)

    return {
        "ok": True,
        "base_image": _obs_to_base64(obs),
        "wrist_image": _obs_to_base64_wrist(obs),
        "step": int(t + 1),
        "done": bool(done),
        "reward": float(reward),
        "instruction": instruction,
        "alignment_ratio": float(alignment_ratio) if alignment_ratio is not None else None,
        "alignment_scores": [(int(s), float(r)) for s, r in _state["alignment_scores"]],
    }


@app.get("/api/state")
def api_state():
    """Get current state."""
    return {
        "env_loaded": _state["env"] is not None,
        "task_id": int(_state["task_id"]) if _state["task_id"] is not None else None,
        "step": int(_state["t"]),
        "done": bool(_state["done"]),
    }


def _run_finetune_thread(req: FinetuneRequest):
    """Run fine-tuning in background thread. Keeps original model, trains a copy, optionally merges."""
    try:
        _state["finetune_status"] = "running"
        _state["finetune_losses"] = []
        _state["finetune_validation_losses"] = []
        _state["finetune_error"] = None
        _state["finetune_total_steps"] = req.num_steps
        _ensure_model()
        if _state["task_id"] is None:
            _state["finetune_status"] = "error"
            _state["finetune_error"] = "Load a task first."
            return
        task_id = _state["task_id"]

        # Task regularization: use selected reference model as base, augment with pseudo-labels
        if req.task_regularization:
            if not req.reference_id:
                _state["finetune_status"] = "error"
                _state["finetune_error"] = "Select a reference model for Task regularization."
                return
            ref_entry = next((r for r in _state["references"] if r["id"] == req.reference_id), None)
            if ref_entry is None:
                _state["finetune_status"] = "error"
                _state["finetune_error"] = f"Reference '{req.reference_id}' not found."
                return
            ref_task_id = req.augment_task_id if req.augment_task_id is not None else ref_entry["task_id"]
            if ref_task_id not in LIBERO_90_TASK_IDS_PROMPTS:
                _state["finetune_status"] = "error"
                _state["finetune_error"] = f"Invalid augment_task_id. Use 0-{len(LIBERO_90_TASK_IDS_PROMPTS) - 1}."
                return
            original_model = ref_entry["model"]
            augment_task_id = ref_task_id
            augment_inference_fn = make_pseudo_label_inference_fn(ref_entry["policy"])
        else:
            original_model = _state["model"]
            augment_task_id = req.augment_task_id
            augment_inference_fn = None

        augment = req.augment or req.task_regularization
        config = _state["config"]
        config = dataclasses.replace(config, batch_size=req.batch_size)
        repo_id = "antoniomari/libero_90"
        with override_create_torch_dataset(
            repo_id=repo_id,
            task_id=task_id,
            mirror_data=True,
            single_episode=req.single_episode,
            augment=augment,
            augment_task_id=augment_task_id,
            augment_inference_fn=augment_inference_fn,
        ):
            data_loader = _data_loader.create_data_loader(
                config, sharding=None, shuffle=True,
            )

        validation_loader = None
        if req.add_validation_set and req.validation_task_id is not None:
            _ensure_task_suite()
            suite = _state["task_suite"]
            val_task_id = req.validation_task_id
            if val_task_id < 0 or val_task_id >= suite.n_tasks:
                _state["finetune_status"] = "error"
                _state["finetune_error"] = f"Invalid validation_task_id. Use 0-{suite.n_tasks - 1}."
                return
            # Use same batch_size as training to avoid JIT recompilation on different shapes
            val_config = dataclasses.replace(config, batch_size=req.batch_size, num_workers=0)
            with override_create_torch_dataset(
                repo_id=repo_id,
                task_id=val_task_id,
                mirror_data=True,
                single_episode=True,
                augment=False,
            ):
                validation_loader = _data_loader.create_data_loader(
                    val_config,
                    sharding=None,
                    shuffle=False,
                    single_epoch=True,
                )

        model_copy = copy_model(original_model, config)

        def _on_step(step: int, loss_val: float) -> None:
            _state["finetune_losses"].append(loss_val)

        def _on_validation(step: int, val_loss: float) -> None:
            _state["finetune_validation_losses"].append({"step": step, "loss": val_loss})

        trained_model, losses, _ = train_model_on_fly(
            model=model_copy,
            training_data_loader=data_loader,
            config=config,
            learning_rate=req.learning_rate,
            num_steps=req.num_steps,
            warmup_steps=req.warmup_steps,
            weight_decay=0.0,
            log_interval=max(1, req.num_steps // 10),
            seed=_state["seed"],
            show_progress_bar=False,
            donate_buffers=False,
            on_step_callback=_on_step,
            validation_data_loader=validation_loader,
            validation_interval=5,
            on_validation_callback=_on_validation if validation_loader else None,
        )

        merging_eps = req.merging_eps if req.merging_eps is not None else 1.0
        if 0.0 <= merging_eps < 1.0:
            final_model = merge_model_parameters(
                trained_model=trained_model,
                original_model=original_model,
                merging_eps=merging_eps,
            )
        else:
            final_model = trained_model

        _state["model"] = final_model
        _state["policy"] = create_policy(
            final_model,
            config,
            CHECKPOINT_DIR,
            rng_seed=_state["seed"],
        )
        _state["finetune_losses"] = losses
        _state["finetune_status"] = "done"
    except Exception as e:
        _state["finetune_status"] = "error"
        _state["finetune_error"] = str(e)
        import traceback
        traceback.print_exc()


@app.post("/api/finetune_start")
def api_finetune_start(req: FinetuneRequest):
    """Start fine-tuning on current task (single-episode demo)."""
    if _state["finetune_status"] == "running":
        raise HTTPException(status_code=400, detail="Fine-tuning already in progress.")
    if _state["task_id"] is None:
        raise HTTPException(status_code=400, detail="Load a task first.")
    _state["finetune_status"] = "running"
    _state["finetune_losses"] = []
    _state["finetune_error"] = None
    _state["finetune_total_steps"] = req.num_steps
    thread = threading.Thread(target=_run_finetune_thread, args=(req,))
    thread.daemon = True
    thread.start()
    return {"ok": True, "status": "running"}


@app.get("/api/finetune_status")
def api_finetune_status():
    """Get fine-tuning status and losses."""
    return {
        "status": _state["finetune_status"],
        "losses": [float(x) for x in _state["finetune_losses"]],
        "validation_losses": list(_state["finetune_validation_losses"]),
        "current_step": int(len(_state["finetune_losses"])),
        "total_steps": int(_state["finetune_total_steps"]),
        "error": _state["finetune_error"],
    }


@app.post("/api/save_as_reference")
def api_save_as_reference():
    """Save current model as reference for Task regularization. Reference is never overwritten by fine-tuning."""
    if _state["model"] is None:
        raise HTTPException(status_code=400, detail="No model loaded. Fine-tune on a task first.")
    if _state["task_id"] is None:
        raise HTTPException(status_code=400, detail="Load a task first.")
    task_id = int(_state["task_id"])
    prompt = LIBERO_90_TASK_IDS_PROMPTS.get(task_id, "unknown")
    name = f"Task {task_id}: {prompt}"
    ref_id = f"ref_{_state['reference_counter']}"
    _state["reference_counter"] += 1
    model_copy = copy_model(_state["model"], _state["config"])
    policy = create_policy(
        model_copy,
        _state["config"],
        CHECKPOINT_DIR,
        rng_seed=_state["seed"],
    )
    ref_entry = {
        "id": ref_id,
        "model": model_copy,
        "policy": policy,
        "task_id": task_id,
        "name": name,
    }
    refs = _state["references"]
    refs.insert(0, ref_entry)
    if len(refs) > MAX_REFERENCES:
        _state["references"] = refs[:MAX_REFERENCES]
    return {
        "ok": True,
        "reference_id": ref_id,
        "reference_name": name,
        "message": f"Reference saved: {name}",
    }


@app.get("/api/reference_status")
def api_reference_status():
    """Get reference models for Task regularization."""
    refs = _state["references"]
    return {
        "references": [
            {"id": r["id"], "name": r["name"], "task_id": r["task_id"]}
            for r in refs
        ],
    }


@app.post("/api/neighbors")
def api_neighbors(req: NeighborsRequest):
    """Fetch nearest neighbors for current observation and return them with images."""
    if _state["env"] is None or _state["obs"] is None:
        raise HTTPException(status_code=400, detail="Load a task first.")
    _ensure_model()
    nn_fetcher = _ensure_nn_fetcher()
    dataset = _ensure_neighbors_dataset()

    obs = _state["obs"]
    instruction = _state["task_instruction"] or ""
    img, wrist_img = _preprocess_images(obs, resize_size=RESIZE_SIZE)
    curr_obs_dict = _build_curr_obs_dict(obs, img, wrist_img, instruction)
    observation = make_observation_from_simulator(_state["policy"], curr_obs_dict)

    if nn_fetcher.normalize_per_modality:
        observation.images["base_0_rgb"] = observation.images["base_0_rgb"][:, :, ::-1, :]
        observation.images["left_wrist_0_rgb"] = observation.images["left_wrist_0_rgb"][:, :, ::-1, :]

    k_fetch = req.k * 5 if req.filter_by_task else req.k
    k_fetch = min(k_fetch, 50)
    distances, indices, _ = nn_fetcher.fetch_neighbors(
        observation=observation,
        use_modalities=["image1", "image2", "text"],
        filter_text_first=True,
        k=k_fetch,
    )

    task_index_filter = None
    if req.filter_by_task and _state["task_id"] is not None:
        task_index_filter = LIBERO_90_TASK_IDS_MAPPING.get(
            _state["task_id"], _state["task_id"]
        )

    neighbors: list[dict] = []
    for i, (idx, sim) in enumerate(zip(indices, distances)):
        if len(neighbors) >= req.k:
            break
        idx = int(idx)
        sample = dataset[idx]
        if task_index_filter is not None:
            sample_ti = _get_sample_task_index(sample)
            if sample_ti is not None and sample_ti != task_index_filter:
                continue
        base_b64, wrist_b64 = _sample_images_to_base64(sample)
        neighbors.append({
            "similarity": float(sim),
            "base_image": base_b64,
            "wrist_image": wrist_b64,
        })

    return {"ok": True, "neighbors": neighbors}


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the main UI."""
    html_path = SCRIPT_DIR / "eval_ui.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>eval_ui.html not found</h1>")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port")
    args = parser.parse_args()

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    import uvicorn
    import socket
    hostname = socket.gethostname()
    username = os.environ.get("USER", "USER")
    print(f"[LIBERO-90 Eval UI] Starting at http://{args.host}:{args.port}")
    print(f"  Open in browser: http://localhost:{args.port}")
    print(f"  To forward from your local machine, run (via login node):")
    print(f"    ssh -L {args.port}:{hostname}:{args.port} {username}@euler.ethz.ch")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
