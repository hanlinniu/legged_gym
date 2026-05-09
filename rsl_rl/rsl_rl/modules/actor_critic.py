# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
from torch.distributions import Normal


class StateHistoryEncoder(nn.Module):
    """1D-CNN over stacked proprioceptive history (RMA-style)."""

    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * channel_size),
            self.activation_fn,
        )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=8, stride=4),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=5, stride=1),
                self.activation_fn,
                nn.Conv1d(in_channels=channel_size, out_channels=channel_size, kernel_size=5, stride=1),
                self.activation_fn,
                nn.Flatten(),
            )
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=4, stride=2),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=2, stride=1),
                self.activation_fn,
                nn.Flatten(),
            )
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=6, stride=2),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=4, stride=2),
                self.activation_fn,
                nn.Flatten(),
            )
        else:
            raise ValueError("tsteps must be 10, 20 or 50")

        self.linear_output = nn.Sequential(
            nn.Linear(channel_size * 3, output_size),
            self.activation_fn,
        )

    def forward(self, obs):
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1]))
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output


class RMAActor(nn.Module):
    """Blind actor: proprio + privilege encoder + history encoder (no scan / terrain encoder)."""

    def __init__(
        self,
        num_prop,
        num_actions,
        actor_hidden_dims,
        priv_encoder_dims,
        num_priv_latent,
        num_priv_explicit,
        num_hist,
        activation,
        tanh_encoder_output=False,
    ) -> None:
        super().__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = []
            priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
            priv_encoder_layers.append(activation)
            for l in range(len(priv_encoder_dims) - 1):
                priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                priv_encoder_layers.append(activation)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)

        actor_layers = []
        actor_layers.append(
            nn.Linear(num_prop + num_priv_explicit + priv_encoder_output_dim, actor_hidden_dims[0])
        )
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

    def forward(self, obs, hist_encoding):
        obs_prop = obs[:, : self.num_prop]
        obs_priv_explicit = obs[:, self.num_prop : self.num_prop + self.num_priv_explicit]
        if hist_encoding:
            latent = self.infer_hist_latent(obs)
        else:
            latent = self.infer_priv_latent(obs)
        backbone_input = torch.cat([obs_prop, obs_priv_explicit, latent], dim=1)
        return self.actor_backbone(backbone_input)

    def infer_priv_latent(self, obs):
        priv = obs[
            :,
            self.num_prop
            + self.num_priv_explicit : self.num_prop
            + self.num_priv_explicit
            + self.num_priv_latent,
        ]
        return self.priv_encoder(priv)

    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist * self.num_prop :]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        ignored = ("scan_encoder_dims", "priv_encoder_dims", "tanh_encoder_output")
        filtered = {k: v for k, v in kwargs.items() if k not in ignored}
        if filtered:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in filtered.keys()])
            )
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, **kwargs):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, **kwargs):
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)


class ActorCriticRMA(nn.Module):
    """PPO actor-critic with RMA-style proprio history and privilege encoders (blind, no scan)."""

    is_recurrent = False

    def __init__(
        self,
        num_prop,
        num_critic_obs,
        num_priv_latent,
        num_priv_explicit,
        num_hist,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        _rma_kw = ("priv_encoder_dims", "tanh_encoder_output", "history_encoding", "scan_encoder_dims")
        extra = [k for k in kwargs.keys() if k not in _rma_kw]
        if extra:
            print(
                "ActorCriticRMA.__init__ got unexpected arguments, which will be ignored: "
                + str(extra)
            )
        super(ActorCriticRMA, self).__init__()

        priv_encoder_dims = kwargs["priv_encoder_dims"]
        tanh_encoder_output = kwargs.get("tanh_encoder_output", False)
        activation = get_activation(activation)
        self.history_encoding = kwargs.get("history_encoding", True)

        self.actor = RMAActor(
            num_prop,
            num_actions,
            actor_hidden_dims,
            priv_encoder_dims,
            num_priv_latent,
            num_priv_explicit,
            num_hist,
            activation,
            tanh_encoder_output=tanh_encoder_output,
        )

        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hist_encoding=None):
        if hist_encoding is None:
            hist_encoding = self.history_encoding
        mean = self.actor(observations, hist_encoding)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, hist_encoding=None, **kwargs):
        self.update_distribution(observations, hist_encoding=hist_encoding)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=None, **kwargs):
        if hist_encoding is None:
            hist_encoding = self.history_encoding
        return self.actor(observations, hist_encoding)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def reset_std(self, std, num_actions, device):
        self.std.data = (std * torch.ones(num_actions, device=device)).data


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
