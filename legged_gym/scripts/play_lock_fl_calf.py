# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
#
# Play script: nominal policy on all joints for --healthy_seconds (default 3 s),
# then hold one DOF (default FL_calf_joint) at a fixed absolute angle (default 45°)
# by overriding only that action dimension; PD gains stay nominal on every joint.

from legged_gym import LEGGED_GYM_ROOT_DIR
import math
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, configure_play_rma_deploy

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
        raise RuntimeError("play_lock_fl_calf expects scalar cfg.control.action_scale.")
    default_q = env.default_dof_pos[0, joint_idx]
    return (target_pos_rad - default_q) / float(scale)


def play_lock_fl_calf(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    configure_play_rma_deploy(train_cfg)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    if hasattr(env_cfg, "fault_curriculum"):
        env_cfg.fault_curriculum.enabled = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if hasattr(env, "fault_gain_scale"):
        env.fault_gain_scale.fill_(1.0)

    if not env.headless and hasattr(env, "lookat_id"):
        print(
            f"[play_lock_fl_calf] Viewer camera follows env index {env.lookat_id} "
            f"(default 1 when num_envs>1). Keys: [ prev env, ] next env, 0–8 jump, F free cam, Space pause."
        )

    joint_name = args.lock_joint_name
    lock_deg = float(args.lock_angle_deg)
    lock_rad = math.radians(lock_deg)
    healthy_s = float(args.healthy_seconds)

    j = _resolve_joint_index(env, joint_name)
    lock_action = _action_for_target_pos(env, j, lock_rad)
    clip_a = float(env.cfg.normalization.clip_actions)
    if abs(lock_action.item()) > clip_a:
        print(
            f"[play_lock_fl_calf] Warning: action needed for {lock_deg}° on {joint_name} "
            f"is |{lock_action.item():.3f}| > clip_actions={clip_a}; target will be clipped in env.step()."
        )

    print(
        f"[play_lock_fl_calf] Healthy play for {healthy_s:.1f} s, then lock {joint_name} "
        f"(index {j}) at {lock_deg}° ({lock_rad:.4f} rad); other joints follow the policy."
    )

    obs = env.get_observations()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

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

    sim_time = 0.0
    locked = False
    lock_announced = False
    for i in range(10 * int(env.max_episode_length)):
        if sim_time >= healthy_s:
            locked = True
        actions = policy(obs.detach())
        if locked:
            actions = actions.clone()
            actions[:, j] = lock_action
        obs, _, rews, dones, infos = env.step(actions.detach())
        sim_time += env.dt
        if locked and not lock_announced:
            lock_announced = True
            print(
                f"[play_lock_fl_calf] Lock active — {joint_name}: target {lock_deg}° "
                f"(action override = {lock_action.item():.4f})."
            )

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
            ri = env.lookat_id if hasattr(env, "lookat_id") else 0
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
    # .pt checkpoint is loaded by runner; set True only to export TorchScript.
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play_lock_fl_calf(args)
