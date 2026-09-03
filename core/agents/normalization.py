"""Observation, return, and action-bound normalization helpers."""

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-5, per_channel=False,
                 norm_only=False, device="cuda"):
        super().__init__()
        self.insize = insize
        self.epsilon = epsilon
        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            self.axis = {
                3: [0, 2, 3],
                2: [0, 2],
                1: [0],
            }[len(insize)]
            size = insize[0]
        else:
            self.axis = [0]
            size = insize
        self.register_buffer("running_mean", torch.zeros(size, dtype=torch.float64, device=device))
        self.register_buffer("running_var", torch.ones(size, dtype=torch.float64, device=device))
        self.register_buffer("count", torch.ones((), dtype=torch.float64, device=device))

    def reset(self, reset_slice=None):
        target = slice(None) if reset_slice is None else reset_slice
        self.running_mean[target].zero_()
        self.running_var[target].fill_(1)
        self.count.fill_(0)

    def update(self, values):
        with torch.no_grad():
            batch_mean = values.mean(self.axis)
            batch_var = values.var(self.axis)
            batch_count = values.size(0)
            self.running_mean, self.running_var, self.count = self._merge(
                self.running_mean,
                self.running_var,
                self.count,
                batch_mean,
                batch_var,
                batch_count,
            )

    @staticmethod
    def _merge(mean, variance, count, batch_mean, batch_variance, batch_count):
        delta = batch_mean - mean
        total = count + batch_count
        new_mean = mean + delta * batch_count / total
        second_moment = (
            variance * count
            + batch_variance * batch_count
            + delta.square() * count * batch_count / total
        )
        return new_mean, second_moment / total, total

    def forward(self, values, denorm=False):
        with torch.no_grad():
            if self.training:
                self.update(values)
            mean, variance = self.running_mean, self.running_var
            if self.per_channel:
                shape = [1, self.insize[0]] + [1] * (len(self.insize) - 1)
                mean = mean.view(shape).expand_as(values)
                variance = variance.view(shape).expand_as(values)
            scale = torch.sqrt(variance.float() + self.epsilon)
            if denorm:
                return torch.clamp(values, -5.0, 5.0) * scale + mean.float()
            normalized = values / scale if self.norm_only else (values - mean.float()) / scale
            return normalized if self.norm_only else torch.clamp(normalized, -5.0, 5.0)


class NormalizeObservation(nn.Module):
    def __init__(self, insize, epsilon=1e-5, per_channel=False,
                 norm_only=False, device="cuda"):
        super().__init__()
        self.running_mean_std = RunningMeanStd(
            insize, epsilon, per_channel, norm_only, device
        )

    def reset(self):
        self.running_mean_std.reset()

    def forward(self, values, denorm=False):
        return self.running_mean_std(values, denorm)


class NormalizeReward(nn.Module):
    def __init__(self, num_envs, insize=1, gamma=0.99, epsilon=1e-8,
                 device="cuda"):
        super().__init__()
        self.return_rms = RunningMeanStd(
            insize, epsilon, norm_only=True, device=device
        )
        self.returns = torch.zeros((num_envs, insize), dtype=torch.float32, device=device)
        self.gamma = gamma if insize == 1 else gamma.repeat(num_envs, 1)
        self.epsilon = epsilon

    def reset(self):
        self.return_rms.reset()
        self.returns.zero_()

    def normalize(self, rewards, dones):
        original_shape = rewards.shape
        rewards = rewards.view(self.returns.shape)
        self.returns = self.returns * self.gamma * (1 - dones).view(-1, 1) + rewards
        self.return_rms.update(self.returns)
        normalized = rewards / torch.sqrt(
            self.return_rms.running_var + self.epsilon
        )
        return normalized.view(original_shape)


class AutoFlatten(nn.Module):
    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, values):
        return torch.flatten(values, start_dim=self.start_dim)


def bound_loss(mean, soft_bound=1.0):
    high = torch.clamp_min(mean - soft_bound, 0).square()
    low = torch.clamp_max(mean + soft_bound, 0).square()
    return (low + high).mean()
