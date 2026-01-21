import logging
class VersionWarningFilter(logging.Filter):
    def filter(self, record):
        # avoid lerobot warning
        return "is in 2.0 format" not in record.getMessage()
logging.getLogger().addFilter(VersionWarningFilter())


import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def collect_data(config, checkpoint_path, data_path, step):
    """Dummy script to collect data using the trained policy. Untested."""

    # TODO: handle imports more gracefully
    import collections
    import pathlib

    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from robosuite.utils.transform_utils import quat2axisangle
    import torch

    from openpi.policies import policy_config

    import shutil
    # needed to load LIBERO initial states
    torch.serialization.add_safe_globals(
        [
            np.core.multiarray._reconstruct,  # noqa
            np.ndarray,
            np.dtype,
            np.dtypes.Float64DType,
        ]
    )

    # TODO: handle config more gracefully
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # noqa
    LIBERO_ENV_RESOLUTION = 256  # noqa
    TASK_SUITE_NAME = "libero_90"  # noqa
    MAX_STEPS = {'libero_spatial': 220, 'libero_object': 280, 'libero_goal': 300, 'libero_10': 520, 'libero_90': 400}  # noqa
    NUM_STEPS_WAIT = 10  # noqa
    NUM_TRIALS_PER_TASK = 40  # noqa
    REPLAN_STEPS = 5  # noqa

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITE_NAME]()
    logging.info(f"Task suite: {TASK_SUITE_NAME}")
    max_steps = MAX_STEPS.get(TASK_SUITE_NAME)
    policy = policy_config.create_trained_policy(config, checkpoint_path)

    if data_path.exists():
        shutil.rmtree(data_path)

    # TODO: read features by main dataset
    # TODO: update repo_id
    collected_dataset = lerobot_dataset.LeRobotDataset.create(
        repo_id=config.data.repo_id,
        root=data_path,
        robot_type="panda",
        fps=10,
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

    total_episodes, total_successes = 0, 0
    # TODO: remove the min() to run on all tasks
    # 6, 20, 27, 29, 32, 55, 56, 59
    for task_id in tqdm.tqdm([29]):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_description = task.language
        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        # TODO: check seeding behavior
        env.seed(42)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(NUM_TRIALS_PER_TASK)):
            logging.info(f"\nTask: {task_description}")
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            logging.info(f"Starting episode {task_episodes+1}...")
            frames = []
            while t < max_steps + NUM_STEPS_WAIT:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                if t < NUM_STEPS_WAIT:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Get preprocessed image
                # IMPORTANT: rotate 180 degrees to match train preprocessing
                og_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                og_wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(og_img, 224, 224))
                wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(og_wrist_img, 224, 224))

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(task_description),
                    }

                    action_chunk = policy.infer(element)["actions"]

                    assert (
                        len(action_chunk) >= REPLAN_STEPS
                    ), f"We want to replan every {REPLAN_STEPS} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[:REPLAN_STEPS])

                action = action_plan.popleft()

                obs, reward, done, info = env.step(action.tolist())

                # optionally zero-out low-norm actions
                action = np.where(np.abs(action) < 0.0011, 0.0, action)

                frames.append(
                    {
                        "image": og_img,
                        "wrist_image": og_wrist_img,
                        "state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ).astype(np.float32),
                        "actions": np.asarray(action, dtype=np.float32),
                        # TODO: maybe only save task decription once per episode?
                        "task": str(task_description),
                    }
                )

                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            if done:
                [collected_dataset.add_frame(f) for f in frames]
                collected_dataset.save_episode()
            task_episodes += 1
            total_episodes += 1

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    wandb.log({"total_success_rate": float(total_successes) / float(total_episodes)}, step=step)
    logging.info(f"Total episodes: {total_episodes}")

    del policy


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

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

    train_state_shape = jax.eval_shape(init, init_rng)
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


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Load model
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
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
    # TODO: handle configs gracefully
    COLLECT_INTERVAL = 100  # noqa
    collected_data_paths = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        print()
        print('Loss:', info['loss'])
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        if step % COLLECT_INTERVAL == 0:
            # checkpoint policy and free up GPU memory; alternatively, offload to RAM
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            checkpoint_manager.wait_until_finished()
            del train_state, train_state_sharding, ptrain_step

            # the new dataset will be saved here with the same repo_id but different path
            new_data_path = epath.Path(config.checkpoint_base_dir) / "data" / str(step)
            collected_data_paths.append(new_data_path)
            collect_data(config, checkpoint_manager._directory / str(step), data_path=new_data_path, step=step)  # noqa

            # now recreate data loader adding the newly collected data
            base_data_config = config.data.create(config.assets_dirs, config.model)
            new_data_config = dataclasses.replace(base_data_config, additional_repo_paths=tuple(collected_data_paths))
            # use create_torch_data_loader directly for now
            data_loader = _data_loader.create_torch_data_loader(
                new_data_config,
                model_config=config.model,
                action_horizon=config.model.action_horizon,
                batch_size=config.batch_size,
                sharding=data_sharding,
                shuffle=True,
                num_batches=None,
                num_workers=config.num_workers,
                seed=config.seed,
                skip_norm_stats=False,
                framework="jax",
            )
            data_iter = iter(data_loader)

            # TODO: vectorize data collection
            # TODO: ensure normalization and prompt saving are correct

            # reload policy from checkpoint
            train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=True)
            train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
            jax.block_until_ready(train_state)
            ptrain_step = jax.jit(
                functools.partial(train_step, config),
                in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
                out_shardings=(train_state_sharding, replicated_sharding),
                donate_argnums=(1,),
            )

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
