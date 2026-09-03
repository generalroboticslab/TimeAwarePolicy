
import torch


def critic_warmup_active(attempted_update, warmup_iters):
    """Return whether an attempted update must train value networks only."""
    return attempted_update < warmup_iters


def actor_critic_loss(policy_loss, value_loss, entropy, bounds_loss, *,
                      critic_only, value_coefficient, entropy_coefficient):
    """Compose a PPO-family loss without actor gradients during critic warmup."""
    loss = value_coefficient * value_loss
    if not critic_only:
        loss = (
            loss
            + policy_loss
            - entropy_coefficient * entropy
            + bounds_loss
        )
    return loss


def successful_episode_indices(terminal_mask, success):
    """Return absolute environment indices for successful terminal episodes."""
    terminal_indices = terminal_mask.nonzero().flatten()
    return terminal_indices[success[terminal_mask].to(torch.bool)]


def update_masked_peak(previous_peak, current_value, active_mask):
    """Update a per-environment peak only where ``active_mask`` is true."""
    active_mask = active_mask.to(dtype=torch.bool)
    return torch.where(
        active_mask & (current_value > previous_peak),
        current_value,
        previous_peak,
    )


def repeated_fixed_config_indices(
        env_ids, *, num_envs, repeats, bank_size):
    """Map environments to an exact repeated enumeration of a fixed bank."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if num_envs % repeats != 0:
        raise ValueError("num_envs must be divisible by repeats")
    config_count = num_envs // repeats
    if config_count > bank_size:
        raise ValueError(
            f"Requested {config_count} configurations from a bank of {bank_size}"
        )
    return env_ids.remainder(config_count)


def compute_staged_time_ratios(
        average_time_ratio, budget_portion, speed_describe, ratio_range,
        *, use_average_speed=False):
    """Build the per-stage temporal ratios for a batch of environments.

    The returned schedule preserves ``average_time_ratio`` when weighted by
    the planned stage durations.  ``speed_describe == 1`` denotes efficient
    stages and ``speed_describe == 0`` denotes stable stages.
    """
    if average_time_ratio.ndim != 1:
        raise ValueError("average_time_ratio must be one-dimensional")
    if len(budget_portion) != len(speed_describe):
        raise ValueError("budget_portion and speed_describe must have equal length")
    if abs(float(sum(budget_portion)) - 1.0) > 1e-6:
        raise ValueError("budget_portion must sum to one")

    fast_portion = sum(
        portion for portion, label in zip(budget_portion, speed_describe)
        if label == 1
    )
    slow_portion = sum(
        portion for portion, label in zip(budget_portion, speed_describe)
        if label == 0
    )
    if fast_portion <= 0 or slow_portion <= 0:
        raise ValueError("the schedule must contain efficient and stable stages")

    ratio_min, ratio_max = ratio_range
    if torch.any(average_time_ratio < ratio_min) or torch.any(
            average_time_ratio > ratio_max):
        raise ValueError("average_time_ratio lies outside ratio_range")

    if use_average_speed:
        fast_time_ratio = slow_time_ratio = average_time_ratio
    else:
        fast_per_slow = slow_portion / fast_portion
        slow_delta = torch.minimum(
            average_time_ratio - ratio_min,
            (ratio_max - average_time_ratio) / fast_per_slow,
        )
        fast_delta = slow_delta * fast_per_slow
        fast_time_ratio = average_time_ratio + fast_delta
        slow_time_ratio = average_time_ratio - slow_delta

    stage_labels = torch.as_tensor(
        speed_describe,
        dtype=torch.bool,
        device=average_time_ratio.device,
    ).unsqueeze(0)
    return torch.where(
        stage_labels,
        fast_time_ratio.unsqueeze(1),
        slow_time_ratio.unsqueeze(1),
    ).expand(-1, len(speed_describe)).clone()


def compute_stage_end_times(real_time_goal, budget_portion):
    """Return the planned end time of every stage."""
    if real_time_goal.ndim != 1:
        raise ValueError("real_time_goal must be one-dimensional")
    portions = torch.as_tensor(
        budget_portion,
        dtype=real_time_goal.dtype,
        device=real_time_goal.device,
    )
    if abs(float(portions.sum().item()) - 1.0) > 1e-6:
        raise ValueError("budget_portion must sum to one")
    end_fractions = torch.cumsum(portions, dim=0)
    return real_time_goal.unsqueeze(1) * end_fractions.unsqueeze(0)


def update_lagrange_multipliers_(multipliers, constraint_estimate, cost_scale,
                                 learning_rate, maximum, *, critic_only):
    """Apply one projected dual step unless the trainer is warming critics."""
    if critic_only:
        return False
    with torch.no_grad():
        multipliers.add_(
            learning_rate * cost_scale * constraint_estimate
        ).clamp_(min=0.0, max=maximum)
    return True


def apply_boundary_semantics(returns, advantages, dones, timeouts, value_bootstrap):
    """Apply rollout-boundary targets and return their training masks.

    A rollout boundary is stored as a state with ``done=True``.  Its sampled
    action is not a valid policy transition.  Under finite-horizon semantics,
    every boundary is a terminal value target.  Under value bootstrap, timeout
    boundaries provide a continuation value to the preceding transition but do
    not have a critic target of their own.
    """
    boundary = dones.bool()
    timeout_boundary = boundary & timeouts.bool()
    terminal_boundary = boundary & ((~timeouts.bool()) | (not value_bootstrap))

    target_mask = terminal_boundary
    while target_mask.ndim < returns.ndim:
        target_mask = target_mask.unsqueeze(-1)

    advantage_mask = boundary
    while advantage_mask.ndim < advantages.ndim:
        advantage_mask = advantage_mask.unsqueeze(-1)

    returns = torch.where(target_mask, torch.zeros_like(returns), returns)
    advantages = torch.where(advantage_mask, torch.zeros_like(advantages), advantages)
    policy_sample_valid = ~boundary
    value_target_valid = ~(timeout_boundary & value_bootstrap)
    return returns, advantages, policy_sample_valid, value_target_valid


def normalized_time_optimal_terminal_reward(remaining_time, maximum_time,
                                            reward_scale):
    """Return ``reward_scale * clamp(remaining_time / maximum_time, 0, 1)``."""
    if maximum_time <= 0:
        raise ValueError("maximum_time must be positive")
    return reward_scale * torch.clamp(remaining_time / maximum_time, 0.0, 1.0)


def paper_time_optimal_terminal_reward(remaining_time, reward_scale):
    """Return the paper reward ``reward_scale * max(T_remaining, 0)``."""
    return reward_scale * torch.clamp(remaining_time, min=0.0)


def rollout_boundary_diagnostics(values, dones, timeouts):
    """Summarize stored boundary rows and reward-critic values at timeouts."""
    boundary = dones.bool()
    timeout_boundary = boundary & timeouts.bool()
    boundary_count = int(boundary.sum().item())
    timeout_count = int(timeout_boundary.sum().item())
    timeout_fraction = timeout_count / boundary_count if boundary_count else 0.0
    timeout_value_mean = (
        float(values[timeout_boundary].mean().item()) if timeout_count else 0.0
    )
    return {
        "boundary_count": boundary_count,
        "timeout_count": timeout_count,
        "timeout_fraction": timeout_fraction,
        "timeout_value_mean": timeout_value_mean,
    }


def normalize_valid_advantages(advantages, valid):
    """Normalize valid policy advantages and keep boundary advantages at zero."""
    normalized = torch.zeros_like(advantages)
    valid_advantages = advantages[valid]
    if valid_advantages.numel() == 0:
        stats_shape = advantages.shape[1:]
        mean = torch.zeros(stats_shape, dtype=advantages.dtype, device=advantages.device)
        std = torch.ones(stats_shape, dtype=advantages.dtype, device=advantages.device)
        return normalized, mean, std

    mean = valid_advantages.mean(dim=0)
    std = valid_advantages.std(dim=0, unbiased=False)
    normalized[valid] = (valid_advantages - mean) / (std + 1e-8)
    return normalized, mean, std


def masked_mean(values, valid, dim=None):
    """Mean over valid samples while preserving any trailing value dimensions."""
    weights = valid.to(device=values.device, dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(values)

    if dim is None:
        denominator = weights.sum().clamp_min(1e-8)
        return (values * weights).sum() / denominator

    denominator = weights.sum(dim=dim).clamp_min(1e-8)
    return (values * weights).sum(dim=dim) / denominator


def flat_grad(loss, parameters, create_graph=False, retain_graph=False):
    """Return a single flat gradient, substituting zeros for unused parameters."""
    parameters = list(parameters)
    gradients = torch.autograd.grad(
        loss,
        parameters,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return torch.cat([
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ])


def get_flat_params(parameters):
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def set_flat_params(parameters, flat_parameters):
    """Copy a flat parameter vector into a module without replacing Parameters."""
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.copy_(flat_parameters[offset:offset + count].view_as(parameter))
            offset += count
    if offset != flat_parameters.numel():
        raise ValueError("Flat parameter vector has the wrong size")


def conjugate_gradient(matrix_vector_product, rhs, max_iterations=10, tolerance=1e-10):
    """Approximately solve ``Ax=rhs`` using only a positive-definite A-vector product."""
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    residual_dot = torch.dot(residual, residual)

    for _ in range(max_iterations):
        product = matrix_vector_product(direction)
        alpha = residual_dot / (torch.dot(direction, product) + 1e-8)
        solution = solution + alpha * direction
        residual = residual - alpha * product
        next_residual_dot = torch.dot(residual, residual)
        if next_residual_dot <= tolerance:
            break
        direction = residual + (next_residual_dot / (residual_dot + 1e-8)) * direction
        residual_dot = next_residual_dot
    return solution


def cpo_search_direction(inv_fisher_reward, inv_fisher_cost, reward_gradient,
                         cost_gradient, constraint_value, max_kl):
    """Compute the single-constraint CPO step in the local trust region.

    The returned direction maximizes the linearized reward while satisfying the
    linearized cost constraint whenever that is possible inside the KL ball.
    If the current policy is too infeasible, it returns the maximal feasible
    recovery step along the natural cost-gradient direction.
    """
    eps = 1e-8
    q = torch.dot(reward_gradient, inv_fisher_reward).clamp_min(eps)
    r = torch.dot(reward_gradient, inv_fisher_cost)
    s = torch.dot(cost_gradient, inv_fisher_cost).clamp_min(eps)

    unconstrained = torch.sqrt(2.0 * max_kl / q) * inv_fisher_reward
    if constraint_value + torch.dot(cost_gradient, unconstrained) <= 0:
        return unconstrained, False

    required_kl = 0.5 * constraint_value.pow(2) / s
    if required_kl >= max_kl:
        recovery = -torch.sqrt(2.0 * max_kl / s) * inv_fisher_cost
        return recovery, True

    tangent = inv_fisher_reward - (r / s) * inv_fisher_cost
    tangent_norm = (q - r.pow(2) / s).clamp_min(eps)
    recovery = -(constraint_value / s) * inv_fisher_cost
    tangent_scale = torch.sqrt(2.0 * (max_kl - required_kl) / tangent_norm)
    return recovery + tangent_scale * tangent, False


class AdaptiveScheduler:
    def __init__(self, kl_threshold = 0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr
    

class LinearScheduler(AdaptiveScheduler):
    def __init__(self, start_lr, min_lr=1e-6, max_steps=1000000, apply_to_entropy=False, **kwargs):
        super().__init__()

        self.start_lr = start_lr
        self.min_lr = min_lr
        self.max_steps = max_steps
        self.apply_to_entropy = apply_to_entropy
        if apply_to_entropy:
            self.start_entropy_coef = kwargs.pop('start_entropy_coef', 0.01)
            self.min_entropy_coef = kwargs.pop('min_entropy_coef', 0.0001)

    def update(self, steps, entropy_coef=0.):
        mul = max(0, self.max_steps - steps)/self.max_steps 
        lr = self.min_lr + (self.start_lr - self.min_lr) * mul
        if self.apply_to_entropy:
            entropy_coef = self.min_entropy_coef + (self.start_entropy_coef - self.min_entropy_coef) * mul

        return lr, entropy_coef    


def linearAmplifier(start_v, max_v, cur_step, max_steps, curr_rate=1):
    mul = min(max_steps, curr_rate * cur_step) / max_steps
    next_v = start_v + (max_v - start_v) * mul
    return next_v
