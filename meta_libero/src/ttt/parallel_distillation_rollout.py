"""Parallel LIBERO environments for on-policy distillation (batched policy inference at replan).

Performance (important):
    ``rollout_num_envs > 1`` batches **JAX policy forwards** (``infer_batch``) across envs that
    replan in the same inner-loop iteration. **MuJoCo / LIBERO** stepping stays **single-threaded
    Python**: each outer-loop tick runs ``env.step`` for active envs **one after another**.

    So wall time is **not** “same as one env” while collecting eight episodes: you still pay
    roughly **one simulator step per active env per tick** (plus preprocessing, per-env metrics).
    What you save vs **eight episodes run back-to-back** is mainly **batched GPU inference** and
    **interleaving** episodes in one process. True “8× sim throughput” would need vectorized /
    multiprocess simulators (not implemented here).

    If rollout feels CPU-bound, try a smaller ``rollout_num_envs``; if GPU inference dominates,
    larger batches can still help utilization.

Profiling:
    Set ``META_LIBERO_PROFILE_ROLLOUT=1`` for one summary line per wave (totals / counts for
    batched student/teacher forwards vs dummy/action ``env.step``).
    Set ``META_LIBERO_PROFILE_ROLLOUT_DETAIL=1`` additionally for every single ``env.step`` and each
    forward pass (verbose; use for short runs).

JAX / XLA batch shape:
    By default (``META_LIBERO_PAD_INFER_BATCH=1``), when fewer than ``b`` envs replan in a tick,
    student/teacher ``infer_batch`` calls are **padded** to ``b`` env slots so the leading batch
    dimension stays fixed across replans (avoids recompilation when ``len(replan_ids)`` changes).
    Set ``META_LIBERO_PAD_INFER_BATCH=0`` to use the old variable batch size (debug only).
"""

from __future__ import annotations

import collections
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from meta_libero.src.rendering import _draw_step_on_frame, make_observation_from_simulator
from meta_libero.src.utils import _get_libero_env, compute_alignment_ratio

# --- Constants (LIBERO / distillation) ---
_RESIZE_SIZE = 224
_REPLAN_STEPS = 5


def _libero_env_resolution(task_suite_name: str) -> int:
    return 224 if task_suite_name == "libero_90" else 256


def _libero_dummy_action() -> list[float]:
    return [0.0] * 6 + [-1.0]


def _rollout_trajectory_id(
    distill_trajectory_id_offset: int, global_episode_idx: int, env_i: int
) -> int:
    return int(distill_trajectory_id_offset) + int(global_episode_idx) + int(env_i)


def _first_student_chunk(st_i: jnp.ndarray) -> jnp.ndarray:
    return st_i[0] if st_i.ndim == 3 else st_i


def _first_teacher_row(te_i: jnp.ndarray) -> jnp.ndarray:
    return te_i[0] if te_i.ndim == 3 else te_i


@dataclass
class _RolloutProfileAcc:
    acc_infer_student_s: float = 0.0
    acc_infer_teacher_s: float = 0.0
    n_infer_batches: int = 0
    acc_step_dummy_s: float = 0.0
    n_step_dummy: int = 0
    acc_step_action_s: float = 0.0
    n_step_action: int = 0


def _infer_off_replan_student_teacher(
    bdl: Any,
    policy: Any,
    teacher_policy: Any,
    curr_obs_dict: dict[str, Any],
    *,
    num_samples: int,
    tgs: int,
    horizon: int,
    action_dim: int,
    cfg_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, float, float]:
    """Fresh student/teacher chunk inference when not on a batched replan step (distillation)."""
    policy._rng, rng_dd = jax.random.split(policy._rng)
    noise_dd = jax.random.normal(
        rng_dd,
        (num_samples, horizon, action_dim),
    )
    student_stacked_dd = bdl._infer_action_chunk_samples(
        policy, curr_obs_dict, noise_dd
    )
    action_chunk_dd = _first_student_chunk(jnp.asarray(student_stacked_dd))
    alignment_ratio_dd = float(
        compute_alignment_ratio(
            policy,
            action_chunk_dd,
            curr_obs_dict,
            noise=noise_dd,
            cfg_weight=cfg_weight,
        )
    )
    if tgs == 1:
        teacher_noise_dd = noise_dd[0:1]
    else:
        policy._rng, rng_tdd = jax.random.split(policy._rng)
        teacher_noise_dd = jax.random.normal(
            rng_tdd,
            (tgs, horizon, action_dim),
        )
    teacher_stacked_dd = bdl._infer_action_chunk_samples(
        teacher_policy, curr_obs_dict, teacher_noise_dd
    )
    teacher_chunk_dd = _first_teacher_row(jnp.asarray(teacher_stacked_dd))
    teacher_var_row_dd = 0.0
    if tgs > 1 and jnp.asarray(teacher_stacked_dd).ndim == 3:
        teacher_var_row_dd = float(
            jnp.mean(jnp.var(jnp.asarray(teacher_stacked_dd), axis=0))
        )
    return (
        jnp.asarray(student_stacked_dd),
        jnp.asarray(teacher_chunk_dd),
        teacher_var_row_dd,
        alignment_ratio_dd,
    )


def _grpo_append_distill_bc_rows(
    bdl: Any,
    policy: Any,
    original_model: Any,
    distillation_examples_out: list[dict[str, Any]],
    obs_m: Any,
    stacked_st: jnp.ndarray,
    teacher_chunk: jnp.ndarray,
    *,
    alignment_ratio: float,
    replan_env_step: int,
    temporal_decay: float,
    teacher_var: float,
    rollout_tid: int,
    grpo_trust_eps: float | None,
    grpo_weight_eps: float,
    grpo_weight: str | None,
) -> None:
    """Append one BC row per group sample (GRPO trust-clamped targets)."""
    _gw = grpo_weight if grpo_weight is not None else "none"
    adv_d = bdl._compute_grpo_advantages(stacked_st, teacher_chunk)
    dists_np = bdl._compute_grpo_dists_np(stacked_st, teacher_chunk)
    w_grpo = (
        bdl._compute_grpo_mean_std_weight(dists_np, eps=grpo_weight_eps)
        if _gw == "mean_std"
        else None
    )
    for i_gr in range(int(stacked_st.shape[0])):
        tgt_gr = bdl._grpo_trust_clamp_target_action(
            stacked_st[i_gr], teacher_chunk, grpo_trust_eps
        )
        distillation_examples_out.append(
            bdl.observation_actions_to_bc_example(
                obs_m,
                tgt_gr,
                model_action_horizon=int(original_model.action_horizon),
                model_action_dim=int(original_model.action_dim),
                alignment_ratio=float(alignment_ratio),
                replan_env_step=int(replan_env_step),
                temporal_decay=float(temporal_decay),
                teacher_chunk_var_mean=float(teacher_var),
                grpo_advantage=float(adv_d[i_gr]),
                grpo_group_weight=w_grpo,
                rollout_trajectory_id=int(rollout_tid),
            )
        )


def _non_grpo_append_distill_merged(
    bdl: Any,
    policy: Any,
    original_model: Any,
    distillation_examples_out: list[dict[str, Any]],
    obs_m: Any,
    teacher_chunk: jnp.ndarray,
    student_first: jnp.ndarray,
    *,
    alpha: float,
    alignment_ratio: float,
    replan_env_step: int,
    temporal_decay: float,
    teacher_var: float,
    rollout_tid: int,
) -> None:
    target_chunk = (1.0 - alpha) * teacher_chunk + alpha * student_first
    distillation_examples_out.append(
        bdl.observation_actions_to_bc_example(
            obs_m,
            target_chunk,
            model_action_horizon=int(original_model.action_horizon),
            model_action_dim=int(original_model.action_dim),
            alignment_ratio=float(alignment_ratio),
            replan_env_step=int(replan_env_step),
            temporal_decay=float(temporal_decay),
            teacher_chunk_var_mean=float(teacher_var),
            rollout_trajectory_id=int(rollout_tid),
        )
    )


def _replan_scatter_metrics_for_env(
    bdl: Any,
    *,
    policy: Any,
    teacher_policy: Any,
    k: int,
    i: int,
    _si: int,
    st_actions: Any,
    te_actions: Any,
    obs_dicts: list[dict[str, Any]],
    student_noises: list[jnp.ndarray],
    teacher_noises: list[jnp.ndarray],
    t: list[int],
    num_samples: int,
    tgs: int,
    cfg_weight: float,
    grpo_weight_eps: float,
    action_plan: list[collections.deque],
    replan_now: list[bool],
    student_stacked_per: list[jnp.ndarray | None],
    teacher_chunk_per: list[jnp.ndarray | None],
    teacher_noise_per: list[jnp.ndarray | None],
    alignment_ratio_per: list[float],
    teacher_alignment_ratio_by_step: list[list[tuple[int, float]]],
    teacher_l2_by_env_step: list[list[tuple[int, float]]],
    alignment_ratio_by_step: list[list[tuple[int, float]]],
    group_sampling_trace: list[list[dict[str, Any]]],
    teacher_group_sampling_trace: list[list[dict[str, Any]]],
    teacher_var_row_per: list[float],
    replan_steps: int,
) -> None:
    """Apply batched infer outputs and metrics for one env in the replan list."""
    replan_now[i] = True
    student_stacked_per[i] = jnp.asarray(st_actions[_si])
    st_i = student_stacked_per[i]
    assert st_i is not None
    action_chunk = _first_student_chunk(st_i)
    te_i = jnp.asarray(te_actions[_si])
    teacher_chunk = _first_teacher_row(te_i)
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
        noise=teacher_noise_per[i],
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

    assert len(action_chunk) >= replan_steps
    action_plan[i].extend(action_chunk[:replan_steps])


def _print_parallel_wave_profile(
    wave: int,
    b: int,
    safety: int,
    inner_loop_wall_s: float,
    prof: _RolloutProfileAcc,
) -> None:
    _sum = (
        prof.acc_infer_student_s
        + prof.acc_infer_teacher_s
        + prof.acc_step_dummy_s
        + prof.acc_step_action_s
    )
    _ms = lambda s: f"{s * 1000:.1f}"
    print(
        "[parallel_rollout profile] "
        f"wave={wave} episodes_this_wave={b} inner_iters={safety} "
        f"wall_inner_s={inner_loop_wall_s:.2f} "
        f"sum_profiled_ms={_ms(_sum)} "
        f"infer_student_ms={_ms(prof.acc_infer_student_s)} "
        f"infer_teacher_ms={_ms(prof.acc_infer_teacher_s)} "
        f"n_replan_pairs={prof.n_infer_batches} "
        f"step_dummy_ms={_ms(prof.acc_step_dummy_s)} n_dummy_steps={prof.n_step_dummy} "
        f"step_action_ms={_ms(prof.acc_step_action_s)} n_action_steps={prof.n_step_action} "
        f"avg_ms_per_dummy={_ms(prof.acc_step_dummy_s / max(1, prof.n_step_dummy))} "
        f"avg_ms_per_action={_ms(prof.acc_step_action_s / max(1, prof.n_step_action))} "
        f"avg_ms_infer_student={_ms(prof.acc_infer_student_s / max(1, prof.n_infer_batches))} "
        f"avg_ms_infer_teacher={_ms(prof.acc_infer_teacher_s / max(1, prof.n_infer_batches))}",
        flush=True,
    )


def _maybe_pad_replan_inputs_to_wave(
    *,
    b_wave: int,
    replan_ids: list[int],
    obs_dicts: list[dict[str, Any]],
    student_noises: list[jnp.ndarray],
    teacher_noises: list[jnp.ndarray],
    curr_cache: list[dict[str, Any] | None],
    policy: Any,
    original_model: Any,
    num_samples: int,
    tgs: int,
    pad_to_wave: bool,
) -> tuple[list[dict[str, Any]], list[jnp.ndarray], list[jnp.ndarray], bool]:
    """Return (obs_list, student_noises_list, teacher_noises_list, padded_infer).

    When ``padded_infer`` is True, ``len(obs_list)==b_wave`` and infer rows align with env indices
    ``0..b_wave-1`` (fixes static batch size for JAX). Otherwise lists match ``replan_ids`` order.
    """
    if not pad_to_wave or len(replan_ids) == b_wave:
        return obs_dicts, student_noises, teacher_noises, False

    obs_by: list[dict[str, Any] | None] = [None] * b_wave
    sn_by: list[jnp.ndarray | None] = [None] * b_wave
    tn_by: list[jnp.ndarray | None] = [None] * b_wave
    for k, ei in enumerate(replan_ids):
        obs_by[ei] = obs_dicts[k]
        sn_by[ei] = student_noises[k]
        tn_by[ei] = teacher_noises[k]

    ref_obs = obs_dicts[0]
    h = int(original_model.action_horizon)
    d = int(original_model.action_dim)
    obs_list: list[dict[str, Any]] = []
    sn_list: list[jnp.ndarray] = []
    tn_list: list[jnp.ndarray] = []
    for j in range(b_wave):
        oj = obs_by[j]
        if oj is not None:
            sj = sn_by[j]
            tnj = tn_by[j]
            assert sj is not None and tnj is not None
            obs_list.append(oj)
            sn_list.append(sj)
            tn_list.append(tnj)
            continue
        pad_obs = curr_cache[j] if curr_cache[j] is not None else ref_obs
        obs_list.append(pad_obs)
        policy._rng, rng_sn = jax.random.split(policy._rng)
        sn_j = jax.random.normal(rng_sn, (num_samples, h, d))
        if tgs == 1:
            tn_j = sn_j[0:1]
        else:
            policy._rng, rng_tn = jax.random.split(policy._rng)
            tn_j = jax.random.normal(rng_tn, (tgs, h, d))
        sn_list.append(sn_j)
        tn_list.append(tn_j)

    return obs_list, sn_list, tn_list, True


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
    """Run ``num_trials`` distillation episodes using up to ``num_envs`` concurrent simulators.

    Simulators advance in one interleaved loop; **policy inference** is batched across replans (see
    module docstring). **Simulation** is sequential per tick, so do not expect wall time equal to a
    single-env rollout when collecting multiple episodes.
    """
    import meta_libero.src.ttt.bundle as bdl

    pad_infer_batch = os.environ.get(
        "META_LIBERO_PAD_INFER_BATCH", "1"
    ).strip().lower() not in ("0", "false", "no")

    VIDEO_OUT_PATH = video_out_path
    libero_res = _libero_env_resolution(task_suite_name)
    num_samples = max(1, int(group_size))
    tgs = max(1, int(teacher_group_size))

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
        # tqdm total=num_trials counts completed episodes; advance when each env finishes, not only
        # after the whole wave (otherwise the bar sits at 0% until the slowest episode ends).
        pbar_episode_logged = [False] * b

        def _pbar_on_env_done(i: int) -> None:
            if not pbar_episode_logged[i]:
                pbar_episode_logged[i] = True
                pbar.update(1)

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
            env, _td = _get_libero_env(task, libero_res, seed + global_episode_idx + i)
            envs.append(env)
            env.seed(seed + global_episode_idx + i)
            env.reset()
            obs.append(env.set_init_state(initial_states[global_episode_idx + i]))

        _prof = os.environ.get("META_LIBERO_PROFILE_ROLLOUT", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "summary",
        )
        _prof_detail = os.environ.get(
            "META_LIBERO_PROFILE_ROLLOUT_DETAIL", ""
        ).strip().lower() in ("1", "true", "yes")
        prof = _RolloutProfileAcc()
        inner_loop_t0 = time.perf_counter()

        safety = 0
        while (not all(done)) and safety < (max_steps + num_steps_wait + 50) * 200:
            safety += 1
            # Wait phase: one dummy step per env per iteration until past wait
            if any((not done[i]) and t[i] < num_steps_wait for i in range(b)):
                for i in range(b):
                    if done[i]:
                        continue
                    if t[i] < num_steps_wait:
                        _t0 = time.perf_counter()
                        obs[i], _r, done[i], _info = envs[i].step(_libero_dummy_action())
                        _dt = time.perf_counter() - _t0
                        if _prof:
                            prof.acc_step_dummy_s += _dt
                            prof.n_step_dummy += 1
                        if _prof_detail:
                            print(
                                f"[parallel_rollout profile] wave={wave} safety={safety} "
                                f"dummy_step env={i} t={t[i]} dt_ms={_dt * 1000:.3f}",
                                flush=True,
                            )
                        t[i] += 1
                        if done[i]:
                            _pbar_on_env_done(i)
                continue

            if all(done):
                break

            for i in range(b):
                if done[i]:
                    continue
                if t[i] >= max_steps + num_steps_wait:
                    done[i] = True
                    _pbar_on_env_done(i)
            if all(done):
                break

            curr_cache: list[dict[str, Any] | None] = [None] * b
            for i in range(b):
                if done[i] or t[i] < num_steps_wait:
                    continue
                img, wrist_img = bdl._preprocess_images(obs[i], resize_size=_RESIZE_SIZE)
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

                ol, snl, tnl, padded_infer = _maybe_pad_replan_inputs_to_wave(
                    b_wave=b,
                    replan_ids=replan_ids,
                    obs_dicts=obs_dicts,
                    student_noises=student_noises,
                    teacher_noises=teacher_noises,
                    curr_cache=curr_cache,
                    policy=policy,
                    original_model=original_model,
                    num_samples=num_samples,
                    tgs=tgs,
                    pad_to_wave=bool(pad_infer_batch and b > 1),
                )
                _br = len(ol) * num_samples
                _brt = len(ol) * tgs
                _t0s = time.perf_counter()
                st_actions = bdl._infer_grouped_batch_multi_env(policy, ol, snl)
                _dts = time.perf_counter() - _t0s
                if _prof:
                    prof.acc_infer_student_s += _dts
                    prof.n_infer_batches += 1
                if _prof_detail:
                    print(
                        f"[parallel_rollout profile] wave={wave} safety={safety} "
                        f"infer_student B={len(replan_ids)} batch_rows={_br} "
                        f"padded={padded_infer} dt_ms={_dts * 1000:.3f}",
                        flush=True,
                    )
                _t0t = time.perf_counter()
                te_actions = bdl._infer_grouped_batch_multi_env(teacher_policy, ol, tnl)
                _dtt = time.perf_counter() - _t0t
                if _prof:
                    prof.acc_infer_teacher_s += _dtt
                if _prof_detail:
                    print(
                        f"[parallel_rollout profile] wave={wave} safety={safety} "
                        f"infer_teacher B={len(replan_ids)} batch_rows={_brt} "
                        f"padded={padded_infer} dt_ms={_dtt * 1000:.3f}",
                        flush=True,
                    )

                for k, i in enumerate(replan_ids):
                    _si = i if padded_infer else k
                    _replan_scatter_metrics_for_env(
                        bdl,
                        policy=policy,
                        teacher_policy=teacher_policy,
                        k=k,
                        i=i,
                        _si=_si,
                        st_actions=st_actions,
                        te_actions=te_actions,
                        obs_dicts=obs_dicts,
                        student_noises=student_noises,
                        teacher_noises=teacher_noises,
                        t=t,
                        num_samples=num_samples,
                        tgs=tgs,
                        cfg_weight=cfg_weight,
                        grpo_weight_eps=grpo_weight_eps,
                        action_plan=action_plan,
                        replan_now=replan_now,
                        student_stacked_per=student_stacked_per,
                        teacher_chunk_per=teacher_chunk_per,
                        teacher_noise_per=teacher_noise_per,
                        alignment_ratio_per=alignment_ratio_per,
                        teacher_alignment_ratio_by_step=teacher_alignment_ratio_by_step,
                        teacher_l2_by_env_step=teacher_l2_by_env_step,
                        alignment_ratio_by_step=alignment_ratio_by_step,
                        group_sampling_trace=group_sampling_trace,
                        teacher_group_sampling_trace=teacher_group_sampling_trace,
                        teacher_var_row_per=teacher_var_row_per,
                        replan_steps=_REPLAN_STEPS,
                    )

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
                tid = _rollout_trajectory_id(
                    distill_trajectory_id_offset, global_episode_idx, i
                )
                if grpo_like:
                    if num_samples < 2:
                        raise ValueError("grpo_like requires group_size >= 2")
                    if replan_now[i]:
                        st_d = jnp.asarray(st_s)
                        assert t_chunk is not None
                        obs_m_d = make_observation_from_simulator(policy, curr_obs_dict)
                        _grpo_append_distill_bc_rows(
                            bdl,
                            policy,
                            original_model,
                            distillation_examples_out,
                            obs_m_d,
                            st_d,
                            t_chunk,
                            alignment_ratio=float(alignment_ratio_per[i]),
                            replan_env_step=int(t[i]),
                            temporal_decay=float(temporal_decay),
                            teacher_var=float(teacher_var_row_per[i]),
                            rollout_tid=tid,
                            grpo_trust_eps=grpo_trust_eps,
                            grpo_weight_eps=grpo_weight_eps,
                            grpo_weight=grpo_weight,
                        )
                    else:
                        st_dd, teacher_chunk_dd, teacher_var_row_dd, alignment_ratio_dd = (
                            _infer_off_replan_student_teacher(
                                bdl,
                                policy,
                                teacher_policy,
                                curr_obs_dict,
                                num_samples=num_samples,
                                tgs=tgs,
                                horizon=int(original_model.action_horizon),
                                action_dim=int(original_model.action_dim),
                                cfg_weight=cfg_weight,
                            )
                        )
                        obs_m_dd = make_observation_from_simulator(
                            policy, curr_obs_dict
                        )
                        _grpo_append_distill_bc_rows(
                            bdl,
                            policy,
                            original_model,
                            distillation_examples_out,
                            obs_m_dd,
                            st_dd,
                            teacher_chunk_dd,
                            alignment_ratio=float(alignment_ratio_dd),
                            replan_env_step=int(t[i]),
                            temporal_decay=float(temporal_decay),
                            teacher_var=float(teacher_var_row_dd),
                            rollout_tid=tid,
                            grpo_trust_eps=grpo_trust_eps,
                            grpo_weight_eps=grpo_weight_eps,
                            grpo_weight=grpo_weight,
                        )
                elif replan_now[i]:
                    sj = jnp.asarray(st_s)
                    stud_first = _first_student_chunk(sj)
                    assert t_chunk is not None
                    obs_m_d = make_observation_from_simulator(policy, curr_obs_dict)
                    _non_grpo_append_distill_merged(
                        bdl,
                        policy,
                        original_model,
                        distillation_examples_out,
                        obs_m_d,
                        t_chunk,
                        stud_first,
                        alpha=alpha,
                        alignment_ratio=float(alignment_ratio_per[i]),
                        replan_env_step=int(t[i]),
                        temporal_decay=float(temporal_decay),
                        teacher_var=float(teacher_var_row_per[i]),
                        rollout_tid=tid,
                    )
                else:
                    st_dd, teacher_chunk_dd, teacher_var_row_dd, alignment_ratio_dd = (
                        _infer_off_replan_student_teacher(
                            bdl,
                            policy,
                            teacher_policy,
                            curr_obs_dict,
                            num_samples=num_samples,
                            tgs=tgs,
                            horizon=int(original_model.action_horizon),
                            action_dim=int(original_model.action_dim),
                            cfg_weight=cfg_weight,
                        )
                    )
                    stud_first = _first_student_chunk(st_dd)
                    obs_m_dd = make_observation_from_simulator(policy, curr_obs_dict)
                    _non_grpo_append_distill_merged(
                        bdl,
                        policy,
                        original_model,
                        distillation_examples_out,
                        obs_m_dd,
                        teacher_chunk_dd,
                        stud_first,
                        alpha=alpha,
                        alignment_ratio=float(alignment_ratio_dd),
                        replan_env_step=int(t[i]),
                        temporal_decay=float(temporal_decay),
                        teacher_var=float(teacher_var_row_dd),
                        rollout_tid=tid,
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
                _t0a = time.perf_counter()
                obs[i], _reward, done[i], _info = envs[i].step(jnp.asarray(action).tolist())
                _dta = time.perf_counter() - _t0a
                if _prof:
                    prof.acc_step_action_s += _dta
                    prof.n_step_action += 1
                if _prof_detail:
                    print(
                        f"[parallel_rollout profile] wave={wave} safety={safety} "
                        f"action_step env={i} t={t[i]} dt_ms={_dta * 1000:.3f}",
                        flush=True,
                    )
                t[i] += 1
                if done[i]:
                    _pbar_on_env_done(i)

        # Rare: inner loop exited on safety cap before all envs finished; keep tqdm consistent.
        if not all(pbar_episode_logged):
            for i in range(b):
                if not pbar_episode_logged[i]:
                    _pbar_on_env_done(i)

        inner_loop_wall_s = time.perf_counter() - inner_loop_t0
        if _prof:
            _print_parallel_wave_profile(
                wave, b, safety, inner_loop_wall_s, prof
            )

        # Task success must come from the env goal check. `done[i]` is also set True when we hit
        # max_steps to stop the inner loop — that must not count as success (single-env path uses
        # env success via last step's done; LIBERO step sets done = _check_success()).
        ep_success_flags: list[bool] = []
        for i in range(b):
            try:
                ep_success_flags.append(bool(envs[i].check_success()))
            except Exception:
                ep_success_flags.append(bool(done[i]))

        for i in range(b):
            try:
                envs[i].close()
            except Exception:
                pass

        # Pack episode metrics for this wave
        for i in range(b):
            ep_idx = global_episode_idx + i
            ep_done = bool(ep_success_flags[i])
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
