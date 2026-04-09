"""Parallel LIBERO environments for on-policy distillation (batched policy inference at replan)."""

from __future__ import annotations

import collections
import pathlib
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from meta_libero.src.rendering import _draw_step_on_frame, make_observation_from_simulator
from meta_libero.src.utils import _get_libero_env, compute_alignment_ratio


def run_parallel_distillation_waves(
    *,
    num_envs: int,
    policy: Any,
    teacher_policy: Any,
    original_model: Any,
    task: Any,
    task_suite_name: str,
    task_id: int,
    task_description: str,
    initial_states: Any,
    num_trials: int,
    num_steps_wait: int,
    max_steps: int,
    seed: int,
    save_video: bool,
    video_out_path: str,
    show_progress_bar: bool,
    distillation_examples_out: list[dict[str, Any]],
    student_action_merge: float,
    group_size: int,
    teacher_group_size: int,
    temporal_decay: float,
    grpo_like: bool,
    grpo_trust_eps: float | None,
    grpo_weight: str | None,
    grpo_weight_eps: float,
    distill_collect_every: int,
    distill_trajectory_id_offset: int,
    write_auxiliary_rollout_pdfs: bool,
    cfg_weight: float,
    after_each_rollout_episode: Callable[[int, int, int], None] | None,
) -> tuple[float, list[dict[str, Any]]]:
    """Run ``num_trials`` distillation episodes using up to ``num_envs`` parallel simulators."""
    import meta_libero.src.ttt.bundle as bdl

    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    RESIZE_SIZE = 224
    REPLAN_STEPS = 5
    LIBERO_ENV_RESOLUTION = 224 if task_suite_name == "libero_90" else 256
    VIDEO_OUT_PATH = video_out_path
    num_samples = max(1, int(group_size))
    tgs = max(1, int(teacher_group_size))
    _gw = grpo_weight if grpo_weight is not None else "none"

    all_episode_metrics: list[dict[str, Any]] = []
    all_teacher_l2_episodes: list[list[tuple[int, float]]] = []
    task_episodes = 0
    task_successes = 0
    global_episode_idx = 0

    pbar = tqdm(
        total=num_trials,
        desc=f"Task {task_id} | parallel envs={num_envs}",
        disable=not show_progress_bar,
    )

    wave = 0
    while global_episode_idx < num_trials:
        b = min(num_envs, num_trials - global_episode_idx)
        envs: list[Any] = []
        obs: list[Any] = []
        done = [False] * b
        t = [0] * b
        action_plan: list[collections.deque] = [collections.deque() for _ in range(b)]
        replay_images: list[list[np.ndarray]] = [[] for _ in range(b)]
        distances_actions: list[list[tuple[int, float]]] = [[] for _ in range(b)]
        similarities: list[list[tuple[int, float]]] = [[] for _ in range(b)]
        alignment_ratio_by_step: list[list[tuple[int, float]]] = [[] for _ in range(b)]
        episode_losses: list[list[list[float]]] = [[] for _ in range(b)]
        episode_test_losses: list[list[list[float]]] = [[] for _ in range(b)]
        neighbor_previews: list[list[dict[str, Any]]] = [[] for _ in range(b)]
        teacher_l2_by_env_step: list[list[tuple[int, float]]] = [[] for _ in range(b)]
        group_sampling_trace: list[list[dict[str, Any]]] = [[] for _ in range(b)]
        teacher_alignment_ratio_by_step: list[list[tuple[int, float]]] = [[] for _ in range(b)]
        teacher_group_sampling_trace: list[list[dict[str, Any]]] = [[] for _ in range(b)]

        for i in range(b):
            env, _td = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed + global_episode_idx + i)
            envs.append(env)
            env.seed(seed + global_episode_idx + i)
            env.reset()
            obs.append(env.set_init_state(initial_states[global_episode_idx + i]))

        safety = 0
        while (not all(done)) and safety < (max_steps + num_steps_wait + 50) * 200:
            safety += 1
            # Wait phase: one dummy step per env per iteration until past wait
            if any((not done[i]) and t[i] < num_steps_wait for i in range(b)):
                for i in range(b):
                    if done[i]:
                        continue
                    if t[i] < num_steps_wait:
                        obs[i], _r, done[i], _info = envs[i].step(LIBERO_DUMMY_ACTION)
                        t[i] += 1
                continue

            if all(done):
                break

            for i in range(b):
                if done[i]:
                    continue
                if t[i] >= max_steps + num_steps_wait:
                    done[i] = True
            if all(done):
                break

            curr_cache: list[dict[str, Any] | None] = [None] * b
            for i in range(b):
                if done[i] or t[i] < num_steps_wait:
                    continue
                img, wrist_img = bdl._preprocess_images(obs[i], resize_size=RESIZE_SIZE)
                replay_images[i].append(_draw_step_on_frame(img, t[i]))
                curr_cache[i] = bdl._build_curr_obs_dict(
                    obs[i], img, wrist_img, task_description
                )

            # Replan batch: envs with empty plan (post-wait)
            replan_ids = [
                i
                for i in range(b)
                if (not done[i])
                and t[i] < max_steps + num_steps_wait
                and len(action_plan[i]) == 0
            ]
            replan_now = [False] * b
            student_stacked_per: list[jnp.ndarray | None] = [None] * b
            teacher_chunk_per: list[jnp.ndarray | None] = [None] * b
            teacher_var_row_per: list[float] = [0.0] * b
            alignment_ratio_per: list[float] = [0.0] * b
            teacher_noise_per: list[jnp.ndarray | None] = [None] * b

            if replan_ids:
                obs_dicts: list[dict[str, Any]] = []
                student_noises: list[jnp.ndarray] = []
                teacher_noises: list[jnp.ndarray] = []
                for i in replan_ids:
                    curr = curr_cache[i]
                    assert curr is not None
                    obs_dicts.append(curr)
                    policy._rng, rng = jax.random.split(policy._rng)
                    student_noises.append(
                        jax.random.normal(
                            rng,
                            (num_samples, original_model.action_horizon, original_model.action_dim),
                        )
                    )
                    policy._rng, rng_teach = jax.random.split(policy._rng)
                    if tgs == 1:
                        tn = student_noises[-1][0:1]
                    else:
                        tn = jax.random.normal(
                            rng_teach,
                            (tgs, original_model.action_horizon, original_model.action_dim),
                        )
                    teacher_noises.append(tn)

                st_actions = bdl._infer_grouped_batch_multi_env(
                    policy, obs_dicts, student_noises
                )
                te_actions = bdl._infer_grouped_batch_multi_env(
                    teacher_policy, obs_dicts, teacher_noises
                )

                for k, i in enumerate(replan_ids):
                    replan_now[i] = True
                    student_stacked_per[i] = jnp.asarray(st_actions[k])
                    st_i = student_stacked_per[i]
                    if st_i.ndim == 3:
                        action_chunk = st_i[0]
                    else:
                        action_chunk = st_i
                    te_i = jnp.asarray(te_actions[k])
                    if te_i.ndim == 3:
                        teacher_chunk = te_i[0]
                    else:
                        teacher_chunk = te_i
                    teacher_chunk_per[i] = teacher_chunk
                    teacher_noise_per[i] = jnp.asarray(teacher_noises[k])

                    alignment_ratio_per[i] = float(
                        compute_alignment_ratio(
                            policy,
                            action_chunk,
                            obs_dicts[k],
                            noise=student_noises[k],
                            cfg_weight=cfg_weight,
                        )
                    )
                    tar = compute_alignment_ratio(
                        teacher_policy,
                        teacher_chunk,
                        obs_dicts[k],
                        noise=teacher_noise_per[k],
                    )
                    teacher_alignment_ratio_by_step[i].append((t[i], float(tar)))
                    l2_ts = bdl.compute_action_distances(teacher_chunk, action_chunk)
                    teacher_l2_by_env_step[i].append((t[i], l2_ts))
                    alignment_ratio_by_step[i].append((t[i], float(alignment_ratio_per[i])))

                    st = jnp.asarray(st_i)
                    tt = jnp.asarray(teacher_chunk)
                    if tt.ndim == 3:
                        tt = tt[0]
                    if num_samples > 1 and st.ndim == 3:
                        diff = st - tt[None, ...]
                        dists_vec = jnp.linalg.norm(diff.reshape(st.shape[0], -1), axis=1)
                        dists_list = [float(v) for v in jax.device_get(dists_vec)]
                        chunk_var_mean = float(jnp.mean(jnp.var(st, axis=0)))
                        dm = float(np.mean(dists_list))
                        ds = float(np.std(dists_list)) + 1e-8
                        norm_list = [float((d - dm) / ds) for d in dists_list]
                        grpo_ms_w = (
                            bdl._compute_grpo_mean_std_weight(dists_list, eps=grpo_weight_eps)
                            if len(dists_list) >= 2
                            else None
                        )
                        adv_arr = bdl._compute_grpo_advantages(st, teacher_chunk)
                        grpo_adv_list = [float(v) for v in adv_arr]
                        group_sampling_trace[i].append(
                            {
                                "env_step": int(t[i]),
                                "chunk_var_mean": chunk_var_mean,
                                "dists": dists_list,
                                "norm_dists": norm_list,
                                "grpo_mean_std_weight": grpo_ms_w,
                                "grpo_advantages": grpo_adv_list,
                            }
                        )
                    else:
                        group_sampling_trace[i].append(
                            {
                                "env_step": int(t[i]),
                                "chunk_var_mean": 0.0,
                                "dists": [],
                                "norm_dists": [],
                                "grpo_mean_std_weight": None,
                                "grpo_advantages": [],
                            }
                        )
                    teacher_var_row = 0.0
                    te_st = jnp.asarray(te_actions[k])
                    if tgs > 1 and te_st.ndim == 3:
                        teacher_var_row = float(jnp.mean(jnp.var(te_st, axis=0)))
                        teacher_group_sampling_trace[i].append(
                            {"env_step": int(t[i]), "chunk_var_mean": teacher_var_row}
                        )
                    else:
                        teacher_group_sampling_trace[i].append(
                            {"env_step": int(t[i]), "chunk_var_mean": 0.0}
                        )
                    teacher_var_row_per[i] = teacher_var_row

                    assert len(action_chunk) >= REPLAN_STEPS
                    action_plan[i].extend(action_chunk[:REPLAN_STEPS])

            # Distillation collection (same timestep as single-env loop)
            for i in range(b):
                if done[i] or t[i] < num_steps_wait:
                    continue
                if not (
                    teacher_policy is not None
                    and distillation_examples_out is not None
                    and t[i] >= num_steps_wait
                    and (t[i] - num_steps_wait) % int(distill_collect_every) == 0
                ):
                    continue
                curr_obs_dict = curr_cache[i]
                if curr_obs_dict is None:
                    continue
                st_s = student_stacked_per[i]
                t_chunk = teacher_chunk_per[i]
                # On replan steps we need cached student/teacher chunks; off-replan GRPO paths
                # run fresh forwards and do not require st_s/t_chunk here.
                if replan_now[i] and (st_s is None or t_chunk is None):
                    continue
                alpha = float(student_action_merge)
                if grpo_like:
                    if num_samples < 2:
                        raise ValueError("grpo_like requires group_size >= 2")
                    if replan_now[i]:
                        st_d = jnp.asarray(st_s)
                        adv_d = bdl._compute_grpo_advantages(st_d, t_chunk)
                        dists_np = bdl._compute_grpo_dists_np(st_d, t_chunk)
                        w_grpo = (
                            bdl._compute_grpo_mean_std_weight(dists_np, eps=grpo_weight_eps)
                            if _gw == "mean_std"
                            else None
                        )
                        obs_m_d = make_observation_from_simulator(policy, curr_obs_dict)
                        for i_gr in range(int(st_d.shape[0])):
                            tgt_gr = bdl._grpo_trust_clamp_target_action(
                                st_d[i_gr], t_chunk, grpo_trust_eps
                            )
                            distillation_examples_out.append(
                                bdl.observation_actions_to_bc_example(
                                    obs_m_d,
                                    tgt_gr,
                                    model_action_horizon=int(original_model.action_horizon),
                                    model_action_dim=int(original_model.action_dim),
                                    alignment_ratio=float(alignment_ratio_per[i]),
                                    replan_env_step=int(t[i]),
                                    temporal_decay=float(temporal_decay),
                                    teacher_chunk_var_mean=float(teacher_var_row_per[i]),
                                    grpo_advantage=float(adv_d[i_gr]),
                                    grpo_group_weight=w_grpo,
                                    rollout_trajectory_id=int(distill_trajectory_id_offset)
                                    + int(global_episode_idx)
                                    + int(i),
                                )
                            )
                    else:
                        policy._rng, rng_dd = jax.random.split(policy._rng)
                        noise_dd = jax.random.normal(
                            rng_dd,
                            (
                                num_samples,
                                original_model.action_horizon,
                                original_model.action_dim,
                            ),
                        )
                        student_stacked_dd = bdl._infer_action_chunk_samples(
                            policy, curr_obs_dict, noise_dd
                        )
                        if student_stacked_dd.ndim == 3:
                            action_chunk_dd = student_stacked_dd[0]
                        else:
                            action_chunk_dd = student_stacked_dd
                        alignment_ratio_dd = compute_alignment_ratio(
                            policy,
                            action_chunk_dd,
                            curr_obs_dict,
                            noise=noise_dd,
                            cfg_weight=cfg_weight,
                        )
                        tgs_dd = tgs
                        if tgs_dd == 1:
                            teacher_noise_dd = noise_dd[0:1]
                        else:
                            policy._rng, rng_tdd = jax.random.split(policy._rng)
                            teacher_noise_dd = jax.random.normal(
                                rng_tdd,
                                (tgs_dd, original_model.action_horizon, original_model.action_dim),
                            )
                        teacher_stacked_dd = bdl._infer_action_chunk_samples(
                            teacher_policy, curr_obs_dict, teacher_noise_dd
                        )
                        if teacher_stacked_dd.ndim == 3:
                            teacher_chunk_dd = teacher_stacked_dd[0]
                        else:
                            teacher_chunk_dd = teacher_stacked_dd
                        teacher_var_row_dd = 0.0
                        if tgs_dd > 1 and teacher_stacked_dd.ndim == 3:
                            teacher_var_row_dd = float(
                                jnp.mean(jnp.var(jnp.asarray(teacher_stacked_dd), axis=0))
                            )
                        st_dd = jnp.asarray(student_stacked_dd)
                        adv_dd = bdl._compute_grpo_advantages(st_dd, teacher_chunk_dd)
                        dists_dd = bdl._compute_grpo_dists_np(st_dd, teacher_chunk_dd)
                        w_grpo_dd = (
                            bdl._compute_grpo_mean_std_weight(dists_dd, eps=grpo_weight_eps)
                            if _gw == "mean_std"
                            else None
                        )
                        obs_m_dd = make_observation_from_simulator(policy, curr_obs_dict)
                        for i_gr in range(int(st_dd.shape[0])):
                            tgt_gr = bdl._grpo_trust_clamp_target_action(
                                st_dd[i_gr], teacher_chunk_dd, grpo_trust_eps
                            )
                            distillation_examples_out.append(
                                bdl.observation_actions_to_bc_example(
                                    obs_m_dd,
                                    tgt_gr,
                                    model_action_horizon=int(original_model.action_horizon),
                                    model_action_dim=int(original_model.action_dim),
                                    alignment_ratio=float(alignment_ratio_dd),
                                    replan_env_step=int(t[i]),
                                    temporal_decay=float(temporal_decay),
                                    teacher_chunk_var_mean=teacher_var_row_dd,
                                    grpo_advantage=float(adv_dd[i_gr]),
                                    grpo_group_weight=w_grpo_dd,
                                    rollout_trajectory_id=int(distill_trajectory_id_offset)
                                    + int(global_episode_idx)
                                    + int(i),
                                )
                            )
                elif replan_now[i]:
                    sj = jnp.asarray(st_s)
                    stud_first = sj[0] if sj.ndim == 3 else sj
                    target_chunk_d = (1.0 - alpha) * t_chunk + alpha * stud_first
                    obs_m_d = make_observation_from_simulator(policy, curr_obs_dict)
                    distillation_examples_out.append(
                        bdl.observation_actions_to_bc_example(
                            obs_m_d,
                            target_chunk_d,
                            model_action_horizon=int(original_model.action_horizon),
                            model_action_dim=int(original_model.action_dim),
                            alignment_ratio=float(alignment_ratio_per[i]),
                            replan_env_step=int(t[i]),
                            temporal_decay=float(temporal_decay),
                            teacher_chunk_var_mean=float(teacher_var_row_per[i]),
                            rollout_trajectory_id=int(distill_trajectory_id_offset)
                            + int(global_episode_idx)
                            + int(i),
                        )
                    )
                else:
                    policy._rng, rng_dd = jax.random.split(policy._rng)
                    noise_dd = jax.random.normal(
                        rng_dd,
                        (
                            num_samples,
                            original_model.action_horizon,
                            original_model.action_dim,
                        ),
                    )
                    student_stacked_dd = bdl._infer_action_chunk_samples(
                        policy, curr_obs_dict, noise_dd
                    )
                    if student_stacked_dd.ndim == 3:
                        action_chunk_dd = student_stacked_dd[0]
                    else:
                        action_chunk_dd = student_stacked_dd
                    alignment_ratio_dd = compute_alignment_ratio(
                        policy,
                        action_chunk_dd,
                        curr_obs_dict,
                        noise=noise_dd,
                        cfg_weight=cfg_weight,
                    )
                    tgs_dd = tgs
                    if tgs_dd == 1:
                        teacher_noise_dd = noise_dd[0:1]
                    else:
                        policy._rng, rng_tdd = jax.random.split(policy._rng)
                        teacher_noise_dd = jax.random.normal(
                            rng_tdd,
                            (tgs_dd, original_model.action_horizon, original_model.action_dim),
                        )
                    teacher_stacked_dd = bdl._infer_action_chunk_samples(
                        teacher_policy, curr_obs_dict, teacher_noise_dd
                    )
                    if teacher_stacked_dd.ndim == 3:
                        teacher_chunk_dd = teacher_stacked_dd[0]
                    else:
                        teacher_chunk_dd = teacher_stacked_dd
                    teacher_var_row_dd = 0.0
                    if tgs_dd > 1 and teacher_stacked_dd.ndim == 3:
                        teacher_var_row_dd = float(
                            jnp.mean(jnp.var(jnp.asarray(teacher_stacked_dd), axis=0))
                        )
                    target_chunk_dd = (1.0 - alpha) * teacher_chunk_dd + alpha * action_chunk_dd
                    obs_m_dd = make_observation_from_simulator(policy, curr_obs_dict)
                    distillation_examples_out.append(
                        bdl.observation_actions_to_bc_example(
                            obs_m_dd,
                            target_chunk_dd,
                            model_action_horizon=int(original_model.action_horizon),
                            model_action_dim=int(original_model.action_dim),
                            alignment_ratio=float(alignment_ratio_dd),
                            replan_env_step=int(t[i]),
                            temporal_decay=float(temporal_decay),
                            teacher_chunk_var_mean=teacher_var_row_dd,
                            rollout_trajectory_id=int(distill_trajectory_id_offset)
                            + int(global_episode_idx)
                            + int(i),
                        )
                    )

            # Environment step
            for i in range(b):
                if done[i]:
                    continue
                if t[i] < num_steps_wait:
                    continue
                if not action_plan[i]:
                    continue
                action = action_plan[i].popleft()
                obs[i], _reward, done[i], _info = envs[i].step(jnp.asarray(action).tolist())
                t[i] += 1
                if done[i]:
                    pass

        for i in range(b):
            try:
                envs[i].close()
            except Exception:
                pass

        # Pack episode metrics for this wave
        for i in range(b):
            ep_idx = global_episode_idx + i
            ep_done = bool(done[i])
            all_teacher_l2_episodes.append(list(teacher_l2_by_env_step[i]))
            all_episode_metrics.append(
                {
                    "episode_idx": ep_idx,
                    "success": ep_done,
                    "distances_actions": list(distances_actions[i]),
                    "similarities": list(similarities[i]),
                    "losses": episode_losses[i],
                    "test_losses": episode_test_losses[i],
                    "num_steps": t[i],
                    "alignment_ratio_by_step": list(alignment_ratio_by_step[i]),
                    "group_sampling_trace": list(group_sampling_trace[i]),
                    "teacher_l2_by_env_step": list(teacher_l2_by_env_step[i]),
                    "teacher_alignment_ratio_by_step": list(teacher_alignment_ratio_by_step[i]),
                    "teacher_group_sampling_trace": list(teacher_group_sampling_trace[i]),
                }
            )
            task_episodes += 1
            if ep_done:
                task_successes += 1
            pbar.update(1)
            if after_each_rollout_episode is not None:
                after_each_rollout_episode(ep_idx, task_episodes, task_successes)
            if save_video:
                bdl._save_rollout_video(
                    save_video=save_video,
                    video_out_path=VIDEO_OUT_PATH,
                    task_id=task_id,
                    task_description=task_description,
                    episode_idx=ep_idx,
                    done=ep_done,
                    replay_images=replay_images[i],
                )
            if write_auxiliary_rollout_pdfs and num_samples > 1 and group_sampling_trace[i]:
                gpdf = pathlib.Path(VIDEO_OUT_PATH) / (
                    f"group_action_sampling_ep{ep_idx:03d}.pdf"
                )
                outp = bdl._plot_group_action_sampling_pdf(
                    group_sampling_trace[i],
                    gpdf,
                    episode_idx=ep_idx,
                    group_size=num_samples,
                )
                if outp is not None and show_progress_bar:
                    print(f"  Saved group sampling metrics plot to {outp}")

        global_episode_idx += b
        wave += 1

    pbar.close()
    success_rate = task_successes / task_episodes if task_episodes > 0 else 0.0
    try:
        bdl.run_evaluation_ttt.last_episode_metrics = all_episode_metrics  # type: ignore[attr-defined]
        bdl.run_evaluation.last_episode_metrics = all_episode_metrics  # type: ignore[attr-defined]
    except Exception:
        pass

    if distillation_examples_out is not None:
        n_d = len(distillation_examples_out)
        _dce = int(distill_collect_every)
        print(
            f"Distillation: collected {n_d} trajectory samples this evaluation run "
            f"({num_trials} episode(s); BC rows when "
            f"(t-num_steps_wait) mod {_dce} == 0 after wait, teacher set)."
        )

    teacher_l2_pdf: pathlib.Path | None = None
    if teacher_policy is not None and write_auxiliary_rollout_pdfs:
        pdf_path = pathlib.Path(VIDEO_OUT_PATH) / (
            f"teacher_student_action_l2_task{task_id}.pdf"
        )
        teacher_l2_pdf = bdl._plot_teacher_student_action_l2_pdf(
            all_teacher_l2_episodes,
            pdf_path,
            task_id=task_id,
        )
        if teacher_l2_pdf is not None:
            print(f"Saved teacher vs student action L2 plot to {teacher_l2_pdf}")
        try:
            bdl.run_evaluation.last_teacher_student_l2_pdf = (  # type: ignore[attr-defined]
                str(teacher_l2_pdf) if teacher_l2_pdf is not None else None
            )
        except Exception:
            pass
    elif teacher_policy is not None:
        try:
            bdl.run_evaluation.last_teacher_student_l2_pdf = None  # type: ignore[attr-defined]
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"Final TTT Results for Task {task_id}:")
    print(f"  Task: {task_description}")
    print(f"  Episodes: {task_episodes}")
    print(f"  Successes: {task_successes}")
    print(f"  Success rate: {success_rate*100:.1f}%")
    print(f"  Total TTT updates: 0")
    print(f"{'='*60}")

    return success_rate, all_episode_metrics
