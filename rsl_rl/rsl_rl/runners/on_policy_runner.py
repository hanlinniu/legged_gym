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

import time
import os
from collections import deque
from copy import copy
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

try:
    import wandb
except ImportError:
    wandb = None

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticRMA, Estimator
from rsl_rl.env import VecEnv


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg.get("estimator")
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])
        policy_name = self.cfg["policy_class_name"]
        self._use_history_encoder = True
        if policy_name == "ActorCriticRMA":
            pc = dict(self.policy_cfg)
            self._use_history_encoder = bool(pc.pop("use_history_encoder", True))
            if not self._use_history_encoder:
                pc["history_encoding"] = False
            actor_critic = actor_critic_class(
                self.env.num_prop,
                num_critic_obs,
                self.env.num_priv_latent,
                self.env.num_priv_explicit,
                self.env.num_hist,
                self.env.num_actions,
                **pc,
            ).to(self.device)
        else:
            actor_critic = actor_critic_class(
                self.env.num_obs,
                num_critic_obs,
                self.env.num_actions,
                **self.policy_cfg,
            ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])
        self._rma_adaptation = False
        self.dagger_update_freq = 0
        if policy_name == "ActorCriticRMA" and self.estimator_cfg is not None:
            ec = self.estimator_cfg
            use_vel_est = bool(ec.get("use_velocity_estimator", True))
            if use_vel_est:
                estimator = Estimator(
                    input_dim=int(ec["num_prop"]),
                    output_dim=int(ec["priv_states_dim"]),
                    hidden_dims=ec["hidden_dims"],
                ).to(self.device)
                self.alg: PPO = alg_class(
                    actor_critic,
                    device=self.device,
                    estimator=estimator,
                    estimator_cfg=ec,
                    **self.alg_cfg,
                )
            else:
                self.alg = alg_class(
                    actor_critic,
                    device=self.device,
                    estimator=None,
                    estimator_cfg=ec,
                    **self.alg_cfg,
                )
            self._rma_adaptation = True
            self.dagger_update_freq = int(self.alg_cfg.get("dagger_update_freq", 20))
        else:
            self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            if self._rma_adaptation and self._use_history_encoder:
                hist_encoding = it % self.dagger_update_freq == 0
            else:
                hist_encoding = False

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if self._rma_adaptation:
                        actions = self.alg.act(obs, critic_obs, hist_encoding, None)
                    else:
                        actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_reward_explr_sum += 0
                        cur_reward_entropy_sum += 0
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_reward_explr_sum[new_ids] = 0
                        cur_reward_entropy_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                start = stop
                self.alg.compute_returns(critic_obs)

            mean_hist_latent_loss = 0.0
            if self._rma_adaptation:
                (
                    mean_value_loss,
                    mean_surrogate_loss,
                    mean_estimator_loss,
                    mean_priv_reg_loss,
                    priv_reg_coef,
                ) = self.alg.update()
                if hist_encoding:
                    mean_hist_latent_loss = self.alg.update_dagger()
            else:
                mean_value_loss, mean_surrogate_loss = self.alg.update()
                mean_estimator_loss = 0.0
                mean_priv_reg_loss = 0.0
                priv_reg_coef = 0.0

            stop = time.time()
            learn_time = stop - start
            mean_disc_loss = 0.0
            mean_disc_acc = 0.0
            entropy_coef = self.alg.entropy_coef
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                wandb_dict['Episode_rew/' + key] = value.item()
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        if self._rma_adaptation:
            self.writer.add_scalar("Loss/estimator", locs["mean_estimator_loss"], locs["it"])
            self.writer.add_scalar("Loss/priv_reg", locs["mean_priv_reg_loss"], locs["it"])
            self.writer.add_scalar("Loss/priv_reg_coef", locs["priv_reg_coef"], locs["it"])
            self.writer.add_scalar("Loss/hist_latent", locs["mean_hist_latent_loss"], locs["it"])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])

        # Weights & Biases (same metric layout as extreme-parkour on_policy_runner.log)
        mean_disc_loss = locs.get("mean_disc_loss", 0.0)
        mean_disc_acc = locs.get("mean_disc_acc", 0.0)
        entropy_coef = locs.get("entropy_coef", self.alg.entropy_coef)
        wandb_dict["Loss/value_function"] = locs["mean_value_loss"]
        wandb_dict["Loss/surrogate"] = locs["mean_surrogate_loss"]
        wandb_dict["Loss/estimator"] = locs["mean_estimator_loss"]
        wandb_dict["Loss/hist_latent_loss"] = locs["mean_hist_latent_loss"]
        wandb_dict["Loss/priv_reg_loss"] = locs["mean_priv_reg_loss"]
        wandb_dict["Loss/priv_ref_lambda"] = locs["priv_reg_coef"]
        wandb_dict["Loss/entropy_coef"] = entropy_coef
        wandb_dict["Loss/learning_rate"] = self.alg.learning_rate
        wandb_dict["Loss/discriminator"] = mean_disc_loss
        wandb_dict["Loss/discriminator_accuracy"] = mean_disc_acc
        wandb_dict["Policy/mean_noise_std"] = mean_std.item()
        wandb_dict["Perf/total_fps"] = fps
        wandb_dict["Perf/collection time"] = locs["collection_time"]
        wandb_dict["Perf/learning_time"] = locs["learn_time"]
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
            mean_r = statistics.mean(locs["rewbuffer"])
            mean_explr = statistics.mean(locs["rew_explr_buffer"])
            mean_entr = statistics.mean(locs["rew_entropy_buffer"])
            wandb_dict["Train/mean_reward"] = mean_r
            wandb_dict["Train/mean_reward_explr"] = mean_explr
            wandb_dict["Train/mean_reward_task"] = mean_r - mean_explr
            wandb_dict["Train/mean_reward_entropy"] = mean_entr
            wandb_dict["Train/mean_episode_length"] = statistics.mean(locs["lenbuffer"])

        if wandb is not None and wandb.run is not None and wandb_dict:
            wandb.log(wandb_dict, step=locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            extra_loss = ""
            if self._rma_adaptation:
                extra_loss += f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                extra_loss += f"""{'Priv reg loss:':>{pad}} {locs['mean_priv_reg_loss']:.4f}\n"""
                extra_loss += f"""{'Priv reg coef:':>{pad}} {locs['priv_reg_coef']:.4f}\n"""
                extra_loss += f"""{'Hist latent (dagger):':>{pad}} {locs['mean_hist_latent_loss']:.4f}\n"""
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{extra_loss}"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        payload = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self._rma_adaptation and self.alg.estimator is not None:
            payload["estimator_state_dict"] = self.alg.estimator.state_dict()
            if self.alg.estimator_optimizer is not None:
                payload["estimator_optimizer_state_dict"] = self.alg.estimator_optimizer.state_dict()
            if self.alg.hist_encoder_optimizer is not None:
                payload["hist_encoder_optimizer_state_dict"] = self.alg.hist_encoder_optimizer.state_dict()
        torch.save(payload, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if self._rma_adaptation and self.alg.estimator is not None and "estimator_state_dict" in loaded_dict:
            self.alg.estimator.load_state_dict(loaded_dict["estimator_state_dict"])
            if load_optimizer and "estimator_optimizer_state_dict" in loaded_dict and self.alg.estimator_optimizer:
                self.alg.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
            if load_optimizer and "hist_encoder_optimizer_state_dict" in loaded_dict and self.alg.hist_encoder_optimizer:
                self.alg.hist_encoder_optimizer.load_state_dict(loaded_dict["hist_encoder_optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        if self._rma_adaptation and getattr(self.alg, "train_with_estimated_states", False):
            if self.alg.estimator is not None:
                self.alg.estimator.eval()
                if device is not None:
                    self.alg.estimator.to(device)
            np_ = self.alg.num_prop
            pd_ = self.alg.priv_states_dim

            def policy(obs):
                with torch.inference_mode():
                    o = obs.clone()
                    o[:, np_ : np_ + pd_] = self.alg.estimator(obs[:, :np_])
                    he = getattr(self.alg.actor_critic, "history_encoding", True)
                    return self.alg.actor_critic.act_inference(o, hist_encoding=he)

            return policy
        return self.alg.actor_critic.act_inference
