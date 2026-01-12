## Training Function
import copy
import dataclasses
import functools
from typing import Any, Iterator, Tuple
import os
import logging

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


class CustomDataLoader(_data_loader.DataLoader):
    """Custom data loader for LIBERO fine-tuning that wraps an iterator."""

    def __init__(self, data_config: _config.DataConfig, iterator: Iterator[Tuple[_model.Observation, _model.Actions]]):
        self._data_config = data_config
        self._iterator = iterator

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self) -> Iterator[Tuple[_model.Observation, _model.Actions]]:
        return iter(self._iterator)


def _batch_to_jax(batch):
    """Ensure that each element of the batch (obs, actions) is a JAX array.

    Recursively converts numpy arrays to JAX arrays while preserving the structure
    of Observation and Actions objects.

    Note: Arrays are kept on CPU initially. JAX will move them to GPU during JIT
    compilation, ensuring only one batch is on GPU at a time.
    """
    if isinstance(batch, tuple) and len(batch) == 2:
        obs, actions = batch

        # Recursively convert numpy arrays to JAX arrays using tree_map
        # This preserves the structure of Observation (nested dicts) and Actions
        # Use jnp.asarray instead of device_put to keep on CPU until JIT moves it
        def _to_jax_array(x):
            if isinstance(x, np.ndarray):
                return jnp.asarray(x)  # Keep on CPU, JIT will move to GPU
            elif isinstance(x, jax.Array):
                # If already on GPU, keep it; otherwise JIT will handle placement
                return x
            else:
                return x  # Preserve non-array types (e.g., None, bool, etc.)

        obs = jax.tree.map(_to_jax_array, obs)
        actions = jax.tree.map(_to_jax_array, actions)
        return (obs, actions)
    else:
        raise ValueError("Batch must be a Tuple of (observation, actions)")




@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    model: _model.BaseModel,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:

    def init(
        rng: at.KeyArrayLike,
        model: _model.BaseModel,
        tx: optax.GradientTransformation,
        partial_params: at.Params | None = None,
    ) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)

        # This line extracts the parameter "state" (i.e., weights and variables) from the model object.
        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    # This line computes and returns the shape/dtype structure of the output of the `init` function
    # when called with `init_rng` (and other required arguments, if any). It does not compute real values,
    # but a PyTree of `ShapeDtypeStruct`s, which is helpful for initialization and partitioning/sharding logic.
    train_state_shape = jax.eval_shape(init, init_rng, model, tx)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding



def main(config: _config.TrainConfig):

    # This code sets up the device mesh and defines sharding specifications for the data and model state.
    # - mesh: Creates a device mesh (for FSDP/sharded training) using the configured list of devices.
    # - data_sharding: Data batches are sharded across devices along the DATA_AXIS.
    # - replicated_sharding: Model state is replicated (not sharded) across all devices.
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")


    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    # This line blocks execution until all pending computations related to `train_state` are finished and
    # its data is available on the host. This ensures that initialization (including any JIT compilation or data transfer)
    # is fully complete before proceeding. It's useful for accurate timing/logging and for catching initialization errors early.
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


def train_model_on_fly(
    model: _model.BaseModel,
    training_set: Iterator[Tuple[_model.Observation, _model.Actions]],
    learning_rate: float = 2.5e-5,
    num_steps: int = 1000,
    batch_size: int = 8,
    warmup_steps: int = 100,
    weight_decay: float = 0.0,
    trainable_filter: Any = None,
    freeze_filter: Any = None,
    log_interval: int = 100,
    seed: int = 42,
) -> _model.BaseModel:
    """
    Train a model on the fly and return a copy of the trained model.

    Args:
        model: The model to train (will be copied internally)
    training_set: Iterator[Tuple[_model.Observation, _model.Actions]],
        learning_rate: Learning rate for optimizer
        num_steps: Number of training steps
        batch_size: Batch size for training
        warmup_steps: Number of warmup steps for learning rate schedule
        weight_decay: Weight decay coefficient
        trainable_filter: Filter for trainable parameters (None = all trainable)
        freeze_filter: Filter for frozen parameters (None = none frozen)
        log_interval: Log training info every N steps
        seed: Random seed

    Returns:
        A copy of the trained model
    """
    # Suppress JAX compilation warnings for cleaner output in notebooks


    # Set the JAX compilation cache directory to avoid recompilation and speed up repeated runs.
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # Setup rng
    rng = jax.random.key(seed)
    train_rng, init_rng = jax.random.split(rng)

    logging.getLogger('absl').setLevel(logging.ERROR)  # Suppress JAX/absl warnings

    # Speed up JIT compilation by reducing XLA autotuning (faster compilation, minimal performance impact)
    # Level 0 = no autotuning (fastest compilation)
    # Level 1 = conservative (default, slower compilation)
    # We use level 0 for faster initial compilation
    if 'XLA_FLAGS' not in os.environ:
        os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
    elif '--xla_gpu_autotune_level' not in os.environ.get('XLA_FLAGS', ''):
        os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_autotune_level=0'

    # Create a deep copy of the model to avoid modifying the original
    # Split and merge to create a copy
    graphdef, state = nnx.split(model)
    # Create new model from the same graphdef but with copied state
    model_copy = nnx.merge(graphdef, state)

    # Initialize optimizer with cosine decay schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate / (warmup_steps + 1),
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_steps,
        end_value=learning_rate * 0.1,
    )

    # Create optimizer
    if weight_decay > 0:
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(),
            optax.add_decayed_weights(weight_decay),
            optax.scale_by_schedule(lr_schedule),
            optax.scale(-1.0),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(),
            optax.scale_by_schedule(lr_schedule),
            optax.scale(-1.0),
        )

    # Get model parameters - aligned with train.py
    params = nnx.state(model_copy)

    # Apply freeze filter if provided - aligned with train.py line 104
    if freeze_filter is not None:
        params = nnx_utils.state_map(
            params, freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
        )

    # Determine trainable filter - aligned with train.py
    # If trainable_filter is provided, use it; otherwise use nnx.Param (all params)
    # If freeze_filter is provided, trainable should exclude frozen (like train.py config.trainable_filter)
    if trainable_filter is not None:
        _trainable_filter_for_init = trainable_filter
    elif freeze_filter is not None:
        # Match train.py: trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))
        _trainable_filter_for_init = nnx.All(nnx.Param, nnx.Not(freeze_filter))
    else:
        # Default: all Params are trainable
        _trainable_filter_for_init = nnx.Param

    # Filter trainable parameters - aligned with train.py line 111
    trainable_params = params.filter(_trainable_filter_for_init)

    # Initialize optimizer state - aligned with train.py line 111
    opt_state = tx.init(trainable_params)

    # Get graphdef for the model (static structure)
    graphdef = nnx.graphdef(model_copy)

    # Create TrainState - mirrors train.py structure
    train_state = training_utils.TrainState(
        step=0,
        params=params,
        model_def=graphdef,
        tx=tx,
        opt_state=opt_state,
        ema_decay=None,  # No EMA for on-the-fly training
        ema_params=None,
    )

    # Initialize RNG
    rng = jax.random.key(seed)

    # Define trainable_filter for training step (must match the one used for init)
    # Use the same filter we used for trainable_params initialization
    _trainable_filter = _trainable_filter_for_init

    # Training step function - mirrors train_step from scripts/train.py
    def train_step(rng, state, batch):
        model = nnx.merge(state.model_def, state.params)
        model.train()

        # Note: @at.typecheck removed for faster compilation (it's expensive during JIT)
        def loss_fn(
            model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
        ):
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            return jnp.mean(chunked_loss)

        train_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch

        # Filter out frozen params - use trainable_filter from closure
        diff_state = nnx.DiffState(0, _trainable_filter)
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

        params = state.params.filter(_trainable_filter)
        # Count the number of trainable parameters after filtering
        def param_count(x):
            if hasattr(x, "value"):
                return jnp.size(x.value)
            return jnp.size(x)
        num_trainable_params = sum(jax.tree_util.tree_leaves(jax.tree_map(param_count, params)))
        jax.debug.print(f"Number of trainable parameters: {num_trainable_params}")

        # Optionally: print or log the number, here we just note it could be accessed if needed.
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Update the model in place and return the new full state
        # new_params is a filtered State (only trainable params), nnx.update will update only those
        nnx.update(model, new_params)
        # Get the full updated state from the model
        new_params = nnx.state(model)

        new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)

        # Convert grads (NNX State) to pure dict for optax.global_norm
        # grads from nnx.value_and_grad returns a State, need to extract .value from Params
        # Use to_pure_dict if available, otherwise extract values
        if hasattr(grads, 'to_pure_dict'):
            grads_dict = grads.to_pure_dict()
        else:
            # Fallback: extract .value from Param objects
            def get_grad_value(x):
                if hasattr(x, 'value'):
                    return x.value
                return x
            grads_dict = jax.tree.map(get_grad_value, grads)

        grad_norm = optax.global_norm(grads_dict)

        info = {
            "loss": loss,
            "grad_norm": grad_norm,
        }
        return new_state, info

    # JIT compile the training step with optimizations
    # donate_argnums=(1,) donates the train_state buffer to avoid copying (like train.py)
    train_step_jit = jax.jit(train_step, donate_argnums=(1,))

    # Training loop
    infos = []
    # Use tqdm.auto for notebook compatibility - it automatically detects notebook environment
    pbar = tqdm(range(num_steps), desc="Training", dynamic_ncols=True, mininterval=0.5, maxinterval=2.0)

    # Initialize iterator
    if callable(training_set):
        data_iter = training_set()
    else:
        data_iter = iter(training_set) if not isinstance(training_set, Iterator) else training_set

    # Warm-up compilation: compile with first batch before starting training loop
    print("Compiling training step (this may take a few minutes)...")
    try:
        warmup_batch = next(data_iter)
        warmup_batch = _batch_to_jax(warmup_batch)
        # Trigger compilation
        _ = train_step_jit(rng, train_state, warmup_batch)
        jax.block_until_ready(_)  # Wait for compilation to finish
        train_state, _ = _
        print("Compilation complete! Starting training...")
    except StopIteration:
        # If iterator is empty, restart it
        if callable(training_set):
            data_iter = training_set()
        else:
            data_iter = iter(training_set)
        warmup_batch = next(data_iter)
        warmup_batch = _batch_to_jax(warmup_batch)
        _ = train_step_jit(rng, train_state, warmup_batch)
        jax.block_until_ready(_)
        train_state, _ = _
        print("Compilation complete! Starting training...")

    for step in pbar:
        # Get batch from training set - aligned with train.py
        try:
            batch = next(data_iter)
        except StopIteration:
            # Restart iterator if exhausted
            if callable(training_set):
                data_iter = training_set()
            else:
                data_iter = iter(training_set)
            batch = next(data_iter)

        # CONVERT batch to jnp arrays
        batch = _batch_to_jax(batch)

        # Training step - aligned with train.py: pass rng, state, batch
        train_state, info = train_step_jit(rng, train_state, batch)

        # Logging - aligned with train.py info structure
        loss_val = float(jax.device_get(info["loss"]))
        grad_norm = float(jax.device_get(info["grad_norm"]))
        infos.append({"loss": loss_val, "grad_norm": grad_norm})

        # Debug: Check if parameters are actually changing (only on first few steps)
        if step < 3:
            # Get a sample parameter to check if it's updating
            sample_param = next(iter(jax.tree.leaves(train_state.params.filter(_trainable_filter))))
            if hasattr(sample_param, 'value'):
                param_val = float(jax.device_get(sample_param.value.flat[0]))
                print(f"  Debug step {step}: sample param value = {param_val:.6f}, grad_norm = {grad_norm:.6f}")

        if step % log_interval == 0 or step == num_steps - 1:
            avg_loss = np.mean([info["loss"] for info in infos[-log_interval:]])
            avg_grad_norm = np.mean([info["grad_norm"] for info in infos[-log_interval:]])
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "grad_norm": f"{avg_grad_norm:.4f}"})
            print(f"Step {step}: loss={avg_loss:.4f}, grad_norm={avg_grad_norm:.4f}")

    print(f"\nTraining completed! Final loss: {infos[-1]['loss']:.4f}")

    # Return a copy of the trained model from final train_state
    trained_model = nnx.merge(train_state.model_def, train_state.params)

    return trained_model
