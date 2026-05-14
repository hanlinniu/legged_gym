# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Play script: run a trained policy while the env alternates timed fault / healthy
# (same mechanism as LeggedRobot distance-timed fault curriculum: one calf held at
# lock_angle_deg via PD, then unlocked). All envs are forced into this mode for testing.
# With a viewer: camera follows env --lookat_id (default 1); [ and ] switch to previous / next env.

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play_fault_health_cycle(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    if not hasattr(env_cfg, "fault_curriculum"):
        raise RuntimeError(
            "This task has no fault_curriculum config (expected LeggedRobot-based env, e.g. go2)."
        )

    fc = env_cfg.fault_curriculum
    fc.enabled = True
    fc.distance_timed_fault = True
    fc.timed_fault_segment_s = float(args.fault_segment_s)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    if not hasattr(env, "fault_distance_timed_active"):
        raise RuntimeError("Environment is not LeggedRobot (missing fault_distance_timed_active).")

    # Camera follow (same keyboard idea as extreme-parkour BaseTask: [ prev, ] next)
    lid = int(args.lookat_id)
    robot_index = min(max(0, lid), env.num_envs - 1)
    env.lookat_id = robot_index
    if not env.headless:
        env.viewer_follow_robot = True
        env.lookat(env.lookat_id)
        if lid < 0 or lid >= env.num_envs:
            print(
                f"[play_fault_health_cycle] --lookat_id {lid} out of range [0, {env.num_envs - 1}]; "
                f"using {robot_index}."
            )
        print(
            f"[play_fault_health_cycle] Camera follows env index {env.lookat_id}. "
            "Press [ for previous env, ] for next."
        )

    seg = float(env.cfg.fault_curriculum.timed_fault_segment_s)
    lock_deg = float(getattr(env.cfg.fault_curriculum, "lock_angle_deg", -120.0))
    print(
        f"[play_fault_health_cycle] Forcing timed fault curriculum on all envs: "
        f"{seg:.1f}s fault (one random calf @ {lock_deg:.0f}°) / {seg:.1f}s healthy, repeating."
    )

    all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.fault_distance_timed_active[:] = True
    env._reset_fault_distance_timed_episode(all_ids)

    j0 = int(env.fault_calf_lock_dof_idx[robot_index].item())
    if j0 >= 0:
        print(
            f"[play_fault_health_cycle] Start: FAULT on DOF {j0} ({env.dof_names[j0]}), "
            f"target {lock_deg:.0f}° via env PD (not action override)."
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
    joint_index = 1
    stop_state_log = 100
    stop_rew_log = env.max_episode_length + 1
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    prev_follow = env.lookat_id
    prev_in_fault = env.fault_timed_in_fault_phase[robot_index].item()
    prev_lock = env.fault_calf_lock_dof_idx[robot_index].item()

    for i in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if not env.headless and env.lookat_id != prev_follow:
            print(f"[play_fault_health_cycle] Camera now follows env index {env.lookat_id}.")
            prev_follow = env.lookat_id
        robot_index = env.lookat_id

        in_fault = env.fault_timed_in_fault_phase[robot_index].item()
        lock_j = env.fault_calf_lock_dof_idx[robot_index].item()
        if in_fault != prev_in_fault or lock_j != prev_lock:
            if in_fault and lock_j >= 0:
                name = env.dof_names[int(lock_j)]
                print(
                    f"[play_fault_health_cycle] t≈{i * env.dt:.2f}s step {i}: FAULT — "
                    f"DOF {lock_j} ({name}) held toward {lock_deg:.0f}° (PD lock)."
                )
            elif not in_fault:
                print(
                    f"[play_fault_health_cycle] t≈{i * env.dt:.2f}s step {i}: HEALTHY — "
                    "no calf lock (policy on all joints)."
                )
            prev_in_fault, prev_lock = in_fault, lock_j

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
    play_fault_health_cycle(args)
