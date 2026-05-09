# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import get_activation


class Estimator(nn.Module):
    """MLP that maps proprioception to privileged explicit states (e.g. base linear velocity)."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=(256, 128, 64),
        activation="elu",
        **kwargs,
    ):
        super(Estimator, self).__init__()
        if kwargs:
            pass
        self.input_dim = input_dim
        self.output_dim = output_dim
        act = get_activation(activation)
        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(act)
        for l in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
            layers.append(act)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.estimator = nn.Sequential(*layers)

    def forward(self, x):
        return self.estimator(x)

    def inference(self, x):
        with torch.no_grad():
            return self.estimator(x)
