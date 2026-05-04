# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
#
# Play script with a single-joint actuator fault: Kp and Kd for one DoF are
# multiplied by x ~ Uniform(0, 1) (same convention as training fault curriculum).
# By default the robot runs with healthy nominal PD for --healthy_seconds, then
# the fault is applied once (simulation time accumulates across the loop).

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def _apply_joint_pd_fault(env, args):
    """Set env.fault_gain_scale so one joint uses x * nominal Kp/Kd; others unchanged."""
    if not hasattr(env, "fault_gain_scale"):
        raise RuntimeError(
            "This task env has no fault_gain_scale (expected LeggedRobot). "
            "Use a legged robot task or extend your env like LeggedRobot."
        )
    nj = env.num_actions
    if args.fault_joint >= 0:
        j = int(min(max(0, args.fault_joint), nj - 1))
    else:
        j = int(torch.randint(0, nj, (1,), device=env.device).item())
    if args.fault_scale >= 0.0:
        x = float(min(1.0, max(0.0, args.fault_scale)))
    else:
        x = float(torch.rand(1, device=env.device).item())
    env.fault_gain_scale.fill_(1.0)
    env.fault_gain_scale[:, j] = x
    dof_name = env.dof_names[j] if j < len(env.dof_names) else f"index_{j}"
    print(f"[play_fault_tolerant] FAULT ON — joint {j} ({dof_name}): Kp/Kd scale x = {x:.4f}")


def play_fault_tolerant(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Same play overrides as play.py
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # Do not run training fault curriculum (would resample / tie to terrain).
    if hasattr(env_cfg, "fault_curriculum"):
        env_cfg.fault_curriculum.enabled = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if hasattr(env, "fault_gain_scale"):
        env.fault_gain_scale.fill_(1.0)

    healthy_s = float(getattr(args, "healthy_seconds", 3.0))
    print(f"[play_fault_tolerant] Nominal PD (healthy) for first {healthy_s:.1f} s, then one joint fault is injected.")

    obs = env.get_observations()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported", "policies")
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 100
    stop_rew_log = env.max_episode_length + 1
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    sim_time = 0.0
    fault_applied = False
    for i in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        sim_time += env.dt
        if not fault_applied and sim_time >= healthy_s:
            _apply_joint_pd_fault(env, args)
            fault_applied = True
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
                    "dof_pos_target": actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
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
    play_fault_tolerant(args)
