"""Reusable neural-network building blocks for TimeAwarePolicy agents."""

import numpy as np
import torch
import torch.nn as nn

from core.agents.normalization import AutoFlatten


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FourierEncoding(nn.Module):
    def __init__(self, in_features, out_features, scale=1.0):
        super().__init__()
        if out_features % 2:
            raise ValueError("out_features must be even")
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        weights = scale * torch.randn(in_features, out_features // 2)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, inputs):
        projected = inputs @ self.weights
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_hidden_layers=None, use_relu=True,
                 output_layernorm=False, output_softplus=False,
                 init_std=1.0, auto_flatten=False, flatten_start_dim=1):
        super().__init__()
        activation = nn.ReLU if use_relu else nn.Tanh
        if isinstance(hidden_size, (list, tuple)):
            widths = list(hidden_size)
        else:
            if num_hidden_layers is None:
                raise ValueError("num_hidden_layers is required for an integer hidden_size")
            widths = [hidden_size] * num_hidden_layers
        if not widths:
            raise ValueError("MLP needs at least one hidden layer")

        modules = [AutoFlatten(flatten_start_dim)] if auto_flatten else []
        previous = input_size
        for width in widths:
            modules.extend([
                layer_init(nn.Linear(previous, width)),
                nn.LayerNorm(width),
                activation(),
            ])
            previous = width
        modules.append(layer_init(nn.Linear(previous, output_size), std=init_std))
        if output_layernorm:
            modules.append(nn.LayerNorm(output_size))
        if output_softplus:
            modules.append(nn.Softplus())
        self.mlp = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.mlp(inputs)
