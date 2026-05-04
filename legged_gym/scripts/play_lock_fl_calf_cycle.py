# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
#
# Same as play_lock_fl_calf.py (joint lock via action override, viewer keys, logging),
# plus fixed body-frame velocity commands (--play_cmd_vx/vy/yaw) after each step.
# Alternates: healthy for cycle_healthy_s (default 3 s), locked for cycle_lock_s
# (default 4 s). When the followed env's episode ends (done), the phase timer resets.

from legged_gym import LEGGED_GYM_ROOT_DIR
import math
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def _resolve_joint_index(env, joint_name: str) -> int:
    names = list(env.dof_names)
    try:
        return names.index(joint_name)
    except ValueError:
        raise RuntimeError(
            f"Joint {joint_name!r} not in env.dof_names ({env.num_dofs} DOFs). "
            f"Use a quadruped URDF that defines this joint (e.g. task go2, a1)."
        ) from None


def _action_for_target_pos(env, joint_idx: int, target_pos_rad: float) -> torch.Tensor:
    """Scalar action so that (action * action_scale + default_dof_pos) == target (P control)."""
    scale = env.cfg.control.action_scale
    if isinstance(scale, dict):
        raise RuntimeError("play_lock_fl_calf_cycle expects scalar cfg.control.action_scale.")
    default_q = env.default_dof_pos[0, joint_idx]
    return (target_pos_rad - default_q) / float(scale)


def _done_for_env(dones: torch.Tensor, env_idx: int) -> bool:
    d = dones.squeeze()
    if d.dim() == 0:
        return bool(d.item())
    return bool(d[env_idx].item())


def _apply_fixed_velocity_commands(env, vx: float, vy: float, yaw: float):
    """Set all envs to the same fixed (vx, vy, yaw_rate) in body frame; requires heading_command=False."""
    env.commands[:, 0] = vx
    env.commands[:, 1] = vy
    env.commands[:, 2] = yaw
    if env.commands.shape[1] > 3:
        env.commands[:, 3] = 0.0


def _refresh_obs_after_command_override(env):
    env.compute_observations()
    co = float(env.cfg.normalization.clip_observations)
    env.obs_buf = torch.clip(env.obs_buf, -co, co)


def play_lock_fl_calf_cycle(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # Fixed velocity commands: direct yaw rate in commands[:,2], no heading mode; avoid periodic resample overwriting.
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1.0e9
    if hasattr(env_cfg, "fault_curriculum"):
        env_cfg.fault_curriculum.enabled = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if hasattr(env, "fault_gain_scale"):
        env.fault_gain_scale.fill_(1.0)

    if not env.headless and hasattr(env, "lookat_id"):
        print(
            f"[play_lock_fl_calf_cycle] Viewer camera follows env index {env.lookat_id} "
            f"(default 1 when num_envs>1). Keys: [ prev env, ] next env, 0–8 jump, F free cam, Space pause."
        )

    joint_name = args.lock_joint_name
    lock_deg = float(args.lock_angle_deg)
    lock_rad = math.radians(lock_deg)
    healthy_s = float(args.cycle_healthy_s)
    lock_s = float(args.cycle_lock_s)
    period = healthy_s + lock_s

    j = _resolve_joint_index(env, joint_name)
    lock_action = _action_for_target_pos(env, j, lock_rad)
    clip_a = float(env.cfg.normalization.clip_actions)
    if abs(lock_action.item()) > clip_a:
        print(
            f"[play_lock_fl_calf_cycle] Warning: action needed for {lock_deg}° on {joint_name} "
            f"is |{lock_action.item():.3f}| > clip_actions={clip_a}; target will be clipped in env.step()."
        )

    cmd_vx = float(args.play_cmd_vx)
    cmd_vy = float(args.play_cmd_vy)
    cmd_yaw = float(args.play_cmd_yaw)
    print(
        f"[play_lock_fl_calf_cycle] Loop: {healthy_s:.1f}s healthy → {lock_s:.1f}s lock {joint_name} "
        f"at {lock_deg}° (repeat). On followed-env episode done, timer resets (next episode starts healthy)."
    )
    print(
        f"[play_lock_fl_calf_cycle] Fixed commands (body frame): vx={cmd_vx}, vy={cmd_vy}, yaw_rate={cmd_yaw} "
        f"(override env resample after each step)."
    )

    obs = env.get_observations()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    _apply_fixed_velocity_commands(env, cmd_vx, cmd_vy, cmd_yaw)
    _refresh_obs_after_command_override(env)
    obs = env.get_observations()

    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported", "policies")
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    logger = Logger(env.dt)
    joint_index = j
    stop_state_log = 100
    stop_rew_log = env.max_episode_length + 1
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    ep_phase_t = 0.0
    prev_locked = None

    for i in range(10 * int(env.max_episode_length)):
        ri = env.lookat_id if hasattr(env, "lookat_id") else 0
        p = ep_phase_t % period
        locked = p >= healthy_s

        if prev_locked is not None and locked != prev_locked:
            state = "LOCK" if locked else "HEALTHY"
            print(f"[play_lock_fl_calf_cycle] {state} — phase_t={ep_phase_t:.3f}s (mod {period:.1f}s)")

        actions = policy(obs.detach())
        if locked:
            actions = actions.clone()
            actions[:, j] = lock_action
        obs, _, rews, dones, infos = env.step(actions.detach())

        _apply_fixed_velocity_commands(env, cmd_vx, cmd_vy, cmd_yaw)
        _refresh_obs_after_command_override(env)
        obs = env.get_observations()

        done_ri = _done_for_env(dones, ri)
        # reset_buf is often 1 before first real step; only treat as episode end after step 0
        if done_ri and i > 0:
            ep_phase_t = 0.0
        elif not done_ri:
            ep_phase_t += env.dt

        prev_locked = locked

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": (
                        actions[ri, joint_index].item() * env.cfg.control.action_scale
                        + env.default_dof_pos[0, joint_index].item()
                    ),
                    "dof_pos": env.dof_pos[ri, joint_index].item(),
                    "dof_vel": env.dof_vel[ri, joint_index].item(),
                    "dof_torque": env.torques[ri, joint_index].item(),
                    "command_x": env.commands[ri, 0].item(),
                    "command_y": env.commands[ri, 1].item(),
                    "command_yaw": env.commands[ri, 2].item(),
                    "base_vel_x": env.base_lin_vel[ri, 0].item(),
                    "base_vel_y": env.base_lin_vel[ri, 1].item(),
                    "base_vel_z": env.base_lin_vel[ri, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[ri, 2].item(),
                    "contact_forces_z": env.contact_forces[ri, env.feet_indices, 2].cpu().numpy(),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play_lock_fl_calf_cycle(args)
