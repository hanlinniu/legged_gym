# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os

import isaacgym
from legged_gym.envs import *
from legged_gym import LEGGED_GYM_ENVS_DIR
from legged_gym.utils import get_args, task_registry
import torch


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
        except ImportError:
            print("Warning: wandb is not installed (`pip install wandb`). Training without wandb.")
            use_wandb = False

    if use_wandb:
        run_name = (args.wandb_name or "").strip() or (
            os.path.basename(ppo_runner.log_dir.rstrip(os.sep)) if ppo_runner.log_dir else args.task
        )
        wandb_kwargs = {
            "project": args.wandb_project,
            "name": run_name,
            "dir": ppo_runner.log_dir or ".",
        }
        ent = (args.wandb_entity or "").strip()
        if ent:
            wandb_kwargs["entity"] = ent
        wandb.init(**wandb_kwargs)
        wandb.config.update(
            {
                "task": args.task,
                "experiment_name": train_cfg.runner.experiment_name,
                "run_name": train_cfg.runner.run_name,
            },
            allow_val_change=True,
        )
        for rel in ("base/legged_robot_config.py", "base/legged_robot.py"):
            p = os.path.join(LEGGED_GYM_ENVS_DIR, rel)
            if os.path.isfile(p):
                wandb.save(p, policy="now")

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


if __name__ == '__main__':
    args = get_args()
    train(args)
