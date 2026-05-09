# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic, ActorCriticRMA
from rsl_rl.storage import RolloutStorage


class PPO:
    def __init__(self, actor_critic, device="cpu", estimator=None, estimator_cfg=None, **kwargs):
        kwargs.pop("dagger_update_freq", None)
        kwargs.pop("priv_reg_coef_schedual_resume", None)

        self.device = device
        self.desired_kl = kwargs.pop("desired_kl", 0.01)
        self.schedule = kwargs.pop("schedule", "fixed")
        self.learning_rate = float(kwargs.pop("learning_rate", 1e-3))

        self.clip_param = float(kwargs.pop("clip_param", 0.2))
        self.num_learning_epochs = int(kwargs.pop("num_learning_epochs", 5))
        self.num_mini_batches = int(kwargs.pop("num_mini_batches", 4))
        self.value_loss_coef = float(kwargs.pop("value_loss_coef", 1.0))
        self.entropy_coef = float(kwargs.pop("entropy_coef", 0.0))
        self.gamma = float(kwargs.pop("gamma", 0.99))
        self.lam = float(kwargs.pop("lam", 0.95))
        self.max_grad_norm = float(kwargs.pop("max_grad_norm", 1.0))
        self.use_clipped_value_loss = bool(kwargs.pop("use_clipped_value_loss", True))

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        self.transition = RolloutStorage.Transition()

        self.estimator = estimator
        self.estimator_cfg = estimator_cfg or {}
        self._rma_adaptation = estimator is not None and isinstance(self.actor_critic, ActorCriticRMA)

        priv_reg_coef_schedual = kwargs.pop("priv_reg_coef_schedual", None)
        if self._rma_adaptation:
            self.priv_reg_coef_schedual = priv_reg_coef_schedual or [0, 0.0, 0, 1]
            self.priv_states_dim = int(self.estimator_cfg["priv_states_dim"])
            self.num_prop = int(self.estimator_cfg["num_prop"])
            self.train_with_estimated_states = bool(self.estimator_cfg.get("train_with_estimated_states", True))
            est_lr = float(self.estimator_cfg.get("learning_rate", self.learning_rate))
            self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=est_lr)
            self.hist_encoder_optimizer = optim.Adam(
                self.actor_critic.actor.history_encoder.parameters(), lr=self.learning_rate
            )
            self.counter = 0
        else:
            self.priv_reg_coef_schedual = priv_reg_coef_schedual
            self.priv_states_dim = None
            self.num_prop = None
            self.train_with_estimated_states = False
            self.estimator_optimizer = None
            self.hist_encoder_optimizer = None
            self.counter = 0

        if kwargs:
            print("PPO.__init__: ignoring unused algorithm keys: " + str(sorted(kwargs.keys())))

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.eval()
        if self.estimator is not None:
            self.estimator.eval()

    def train_mode(self):
        self.actor_critic.train()
        if self.estimator is not None:
            self.estimator.train()

    def act(self, obs, critic_obs, hist_encoding=False, info=None):
        del info  # reserved; blind training does not use depth / scan bundles
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        if self._rma_adaptation and self.train_with_estimated_states:
            obs_est = obs.clone()
            priv_hat = self.estimator(obs_est[:, : self.num_prop])
            obs_est[:, self.num_prop : self.num_prop + self.priv_states_dim] = priv_hat
            self.transition.actions = self.actor_critic.act(obs_est, hist_encoding=hist_encoding).detach()
        elif isinstance(self.actor_critic, ActorCriticRMA):
            he = hist_encoding if hist_encoding is not None else getattr(
                self.actor_critic, "history_encoding", False
            )
            self.transition.actions = self.actor_critic.act(obs, hist_encoding=he).detach()
        else:
            self.transition.actions = self.actor_critic.act(obs).detach()

        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        rewards_total = rewards.clone()
        self.transition.rewards = rewards_total.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
        return rewards_total

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update_counter(self):
        self.counter += 1

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_estimator_loss = 0.0
        mean_priv_reg_loss = 0.0
        priv_reg_coef = 0.0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            if self.actor_critic.is_recurrent:
                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(
                    critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
                )
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                            / (2.0 * torch.square(sigma_batch))
                            - 0.5,
                            axis=-1,
                        )
                        kl_mean = torch.mean(kl)
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                continue

            if self._rma_adaptation:
                self.actor_critic.update_distribution(
                    obs_batch, hist_encoding=self.actor_critic.history_encoding
                )
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
                with torch.inference_mode():
                    hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
                priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
                sched = self.priv_reg_coef_schedual
                priv_reg_stage = min(max((self.counter - sched[2]), 0) / max(sched[3], 1e-8), 1.0)
                priv_reg_coef = priv_reg_stage * (sched[1] - sched[0]) + sched[0]

                priv_states_predicted = self.estimator(obs_batch[:, : self.num_prop])
                estimator_loss = (
                    priv_states_predicted
                    - obs_batch[:, self.num_prop : self.num_prop + self.priv_states_dim]
                ).pow(2).mean()
                self.estimator_optimizer.zero_grad()
                estimator_loss.backward()
                nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
                self.estimator_optimizer.step()

                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                            + (
                                torch.square(old_sigma_batch)
                                + torch.square(old_mu_batch - mu_batch)
                            )
                            / (2.0 * torch.square(sigma_batch))
                            - 0.5,
                            axis=-1,
                        )
                        kl_mean = torch.mean(kl)
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + priv_reg_coef * priv_reg_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimator_loss += estimator_loss.item()
                mean_priv_reg_loss += priv_reg_loss.item()
                continue

            # Flat ActorCritic (non-recurrent, no RMA adaptation)
            if isinstance(self.actor_critic, ActorCriticRMA):
                self.actor_critic.update_distribution(
                    obs_batch, hist_encoding=getattr(self.actor_critic, "history_encoding", True)
                )
            else:
                self.actor_critic.act(obs_batch)

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()
        self.update_counter()

        if self._rma_adaptation:
            mean_estimator_loss /= num_updates
            mean_priv_reg_loss /= num_updates
            return mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_priv_reg_loss, priv_reg_coef

        return mean_value_loss, mean_surrogate_loss

    def update_dagger(self):
        mean_hist_latent_loss = 0.0
        if not self._rma_adaptation:
            return mean_hist_latent_loss

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            with torch.inference_mode():
                priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
            hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm
            )
            self.hist_encoder_optimizer.step()
            mean_hist_latent_loss += hist_latent_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss
