"""PPO-family and CPO policy-update implementations."""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import wandb

from core.agents.normalization import bound_loss
from core.agents.utils import (
    actor_critic_loss,
    apply_boundary_semantics,
    conjugate_gradient,
    cpo_search_direction,
    critic_warmup_active,
    flat_grad,
    get_flat_params,
    masked_mean,
    normalize_valid_advantages,
    set_flat_params,
    update_lagrange_multipliers_,
)


class PolicyUpdateMixin:
    def update_policy(self, returns, advantages, returns_c=None, advantages_c=None, initial_lstm_state=None):
        """Update policy using PPO or P3O."""
        if self.args.cmdp_method == "cpo":
            return self.update_policy_cpo(
                returns, advantages, returns_c, advantages_c, initial_lstm_state
            )

        # Flatten batches
        obs_shape = self.envs.obs_space.shape
        state_shape = self.envs.state_space.shape
        act_shape = self.envs.act_space.shape if not self.args.meta_rl else (2,)

        b_obs = self.obs.reshape((-1,) + obs_shape)
        b_states = self.states.reshape((-1,) + state_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + act_shape)
        b_dones = self.dones.reshape(-1)
        b_timeouts = self.timeouts.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        b_returns, b_advantages, policy_sample_valid, value_target_valid = apply_boundary_semantics(
            b_returns,
            b_advantages,
            b_dones,
            b_timeouts,
            self.args.value_bootstrap,
        )

        if self.args.use_cost:
            b_advantages_c = advantages_c.reshape(-1, self.args.num_cost)
            b_returns_c = returns_c.reshape(-1, self.args.num_cost)
            b_values_c = self.values_c.reshape(-1, self.args.num_cost)
            b_returns_c, b_advantages_c, _, _ = apply_boundary_semantics(
                b_returns_c,
                b_advantages_c,
                b_dones,
                b_timeouts,
                self.args.value_bootstrap,
            )

        # Boundary actions are still sampled to drive the environment reset, but
        # they are excluded from normalization and have zero policy advantage.
        b_advantages, _, _ = normalize_valid_advantages(b_advantages, policy_sample_valid)
        if self.args.use_cost:
            b_advantages_c, b_adv_c_mean, b_adv_c_std = normalize_valid_advantages(
                b_advantages_c, policy_sample_valid
            )
            constraint_estimate = (
                (1.0 - self.c_gamma.view(-1)) * b_returns_c.mean(dim=0) + b_adv_c_mean
            ) / (b_adv_c_std + 1e-8)

        # Prepare indices
        if self.args.use_lstm:
            envsperbatch = self.args.num_envs // self.args.num_minibatches
            envinds = np.arange(self.args.num_envs)
            flatinds = np.arange(self.args.batch_size).reshape(self.args.num_steps, self.args.num_envs)
            end_idx = self.args.num_envs
            step_num = envsperbatch
        else:
            b_inds = np.arange(self.args.batch_size)
            end_idx = self.args.batch_size
            step_num = self.args.minibatch_size

        # Save previous parameters for KL divergence check
        if self.args.target_kl is not None:
            agent_params_store = deepcopy(self.agent.state_dict())
            optim_params_store = deepcopy(self.optimizer.state_dict())

        policy_diverged = False

        for epoch in range(self.args.update_epochs):
            if self.args.use_lstm:
                np.random.shuffle(envinds)
            else:
                np.random.shuffle(b_inds)

            for start in range(0, end_idx, step_num):
                end = start + step_num

                if self.args.use_lstm:
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:, mbenvinds].ravel()
                    _, mu, newlogprob, entropy, newvalue, _, newvalue_c = self.agent.get_action_and_value(
                        b_obs[mb_inds],
                        b_states[mb_inds],
                        (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds],
                         initial_lstm_state[2][:, mbenvinds], initial_lstm_state[3][:, mbenvinds]),
                        b_dones[mb_inds],
                        b_actions[mb_inds],
                    )
                else:
                    mb_inds = b_inds[start:end]
                    _, mu, newlogprob, entropy, newvalue, newvalue_c = self.agent.get_action_and_value(
                        b_obs[mb_inds],
                        b_states[mb_inds],
                        b_actions[mb_inds]
                    )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                mb_policy_valid = policy_sample_valid[mb_inds]
                mb_value_valid = value_target_valid[mb_inds]

                with torch.no_grad():
                    old_approx_kl = masked_mean(-logratio, mb_policy_valid)
                    approx_kl = masked_mean((ratio - 1) - logratio, mb_policy_valid)
                    policy_diverged = self.args.target_kl is not None and approx_kl > self.args.target_kl

                # Policy loss
                clipped_ratio = torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * clipped_ratio
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * masked_mean(v_loss_max, mb_value_valid)
                else:
                    v_loss = 0.5 * masked_mean(
                        (newvalue - b_returns[mb_inds]) ** 2, mb_value_valid
                    )

                # Cost loss
                if self.args.use_cost:
                    mb_advantages_c = b_advantages_c[mb_inds]
                    cost_loss1 = mb_advantages_c * ratio.view(-1, 1)
                    cost_loss2 = mb_advantages_c * clipped_ratio.view(-1, 1)
                    L_clip_c = torch.max(cost_loss1, cost_loss2).mean(dim=0)

                    batch_cost_ret = (1.0 - self.c_gamma) * b_returns_c[mb_inds].mean(dim=0)
                    batch_cost_ret = (batch_cost_ret + b_adv_c_mean) / (b_adv_c_std + 1e-8)
                    L_viol_vector = L_clip_c + batch_cost_ret
                    if self.args.cmdp_method == "ppo_lagrangian":
                        L_viol = (
                            self.c_scale.view(-1)
                            * self.lagrange_multipliers.detach()
                            * L_viol_vector
                        ).sum()
                    else:
                        L_viol = (
                            self.c_scale.view(-1)
                            * torch.clamp(L_viol_vector, min=0.0)
                        ).sum()
                    pg_loss += L_viol

                    # Cost value loss
                    newvalue_c = newvalue_c.view(-1, self.args.num_cost)
                    if self.args.clip_vloss:
                        v_loss_unclipped_c = (newvalue_c - b_returns_c[mb_inds]) ** 2
                        v_clipped_c = b_values_c[mb_inds] + torch.clamp(
                            newvalue_c - b_values_c[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped_c = (v_clipped_c - b_returns_c[mb_inds]) ** 2
                        v_loss_max_c = torch.max(v_loss_unclipped_c, v_loss_clipped_c)
                        v_loss_c = 0.5 * masked_mean(
                            v_loss_max_c, mb_value_valid, dim=0
                        ).sum()
                    else:
                        v_loss_c = 0.5 * masked_mean(
                            (newvalue_c - b_returns_c[mb_inds]) ** 2,
                            mb_value_valid,
                            dim=0,
                        ).sum()
                    v_loss += v_loss_c

                entropy_loss = masked_mean(entropy, mb_policy_valid)
                critic_only = critic_warmup_active(
                    self.attempted_update_iter - 1, self.args.warmup_iters
                )
                bounds_loss = v_loss.new_zeros(())
                if not self.args.beta and mb_policy_valid.any():
                    bounds_loss = self.args.bounds_loss_coef * bound_loss(
                        mu[mb_policy_valid], soft_bound=1.
                    )
                loss = actor_critic_loss(
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    bounds_loss,
                    critic_only=critic_only,
                    value_coefficient=self.args.vf_coef,
                    entropy_coefficient=self.cur_ent,
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.args.max_grad_norm)
                nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.args.max_grad_norm)
                if self.args.use_cost:
                    nn.utils.clip_grad_norm_(self.agent.critic_inst.parameters(), self.args.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.agent.critic_t.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

                if policy_diverged:
                    break

                if self.args.anneal_lr:
                    if self.args.scheduler == 'adapt':
                        new_lr = self.lr_scheduler.update(self.optimizer.param_groups[0]["lr"], approx_kl)
                    else:
                        new_lr, _ = self.lr_scheduler.update(self.global_step)
                    self.optimizer.param_groups[0]["lr"] = new_lr

            if policy_diverged:
                break

        if policy_diverged:
            self.agent.load_state_dict(agent_params_store)
            self.optimizer.load_state_dict(optim_params_store)
            self.skipped_update_iter += 1

            if self.args.saving and self.args.wandb:
                wandb.log({
                    'debug/skipped_update_iter': self.skipped_update_iter,
                    'debug/skipped_kl': approx_kl.item(),
                    'debug/skipped_adv': mb_advantages.mean().item(),
                    'debug/skipped_ratio': ratio.mean().item(),
                    'debug/skipped_entropy': entropy_loss.item(),
                })
            return True

        if (
            self.args.use_cost
            and self.args.cmdp_method == "ppo_lagrangian"
        ):
            update_lagrange_multipliers_(
                self.lagrange_multipliers,
                constraint_estimate,
                self.c_scale.view(-1),
                self.args.lagrangian_lr,
                self.args.lagrangian_max,
                critic_only=critic_warmup_active(
                    self.attempted_update_iter - 1, self.args.warmup_iters
                ),
            )

        # Compute explained variance
        y_pred = b_values.to(torch.float32).cpu().numpy()
        y_true = b_returns.to(torch.float32).cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log metrics
        self._log_training_metrics(pg_loss, v_loss, entropy_loss, approx_kl, mb_advantages, explained_var,
                                   v_loss_c if self.args.use_cost else None,
                                   L_viol if self.args.use_cost else None)

        return False


    def update_policy_cpo(self, returns, advantages, returns_c, advantages_c, initial_lstm_state=None):
        """CPO update with the same rollout, cost signal, and value models as N-P3O.

        CubeStack uses one active constraint (``c_scale=[0, 1]``).  The actor
        receives a standard single-constraint CPO natural-gradient step; the
        reward and cost critics retain the same clipped/unclipped regression
        configuration used by the PPO-family updates.
        """
        if self.args.use_lstm:
            raise NotImplementedError("CPO is only implemented for the feed-forward policy")

        obs_shape = self.envs.obs_space.shape
        state_shape = self.envs.state_space.shape
        act_shape = self.envs.act_space.shape if not self.args.meta_rl else (2,)
        b_obs = self.obs.reshape((-1,) + obs_shape)
        b_states = self.states.reshape((-1,) + state_shape)
        b_actions = self.actions.reshape((-1,) + act_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_dones = self.dones.reshape(-1)
        b_timeouts = self.timeouts.reshape(-1)
        b_values = self.values.reshape(-1)
        b_values_c = self.values_c.reshape(-1, self.args.num_cost)

        b_returns, b_advantages, policy_sample_valid, value_target_valid = apply_boundary_semantics(
            returns.reshape(-1),
            advantages.reshape(-1),
            b_dones,
            b_timeouts,
            self.args.value_bootstrap,
        )
        b_returns_c, b_advantages_c, _, _ = apply_boundary_semantics(
            returns_c.reshape(-1, self.args.num_cost),
            advantages_c.reshape(-1, self.args.num_cost),
            b_dones,
            b_timeouts,
            self.args.value_bootstrap,
        )
        b_advantages, _, _ = normalize_valid_advantages(b_advantages, policy_sample_valid)
        b_advantages_c, b_adv_c_mean, b_adv_c_std = normalize_valid_advantages(
            b_advantages_c, policy_sample_valid
        )

        active_constraints = torch.nonzero(self.c_scale.view(-1) > 0, as_tuple=False).flatten()
        if active_constraints.numel() != 1:
            raise ValueError(
                "CPO requires exactly one active constraint; set one c_scale entry > 0"
            )
        active_cost = int(active_constraints.item())
        constraint_baseline = (
            (1.0 - self.c_gamma.view(-1)[active_cost]) * b_returns_c[:, active_cost].mean()
            + b_adv_c_mean[active_cost]
        ) / (b_adv_c_std[active_cost] + 1e-8)
        constraint_baseline = constraint_baseline.detach()

        valid_indices = torch.nonzero(policy_sample_valid, as_tuple=False).flatten()
        if valid_indices.numel() > self.args.cpo_batch_size:
            permutation = torch.randperm(valid_indices.numel(), device=self.device)
            actor_indices = valid_indices[permutation[:self.args.cpo_batch_size]]
        else:
            actor_indices = valid_indices
        if actor_indices.numel() == 0:
            raise RuntimeError("CPO rollout contains no valid policy transitions")

        actor_parameters = [
            parameter
            for name, parameter in self.agent.named_parameters()
            if (name.startswith("actor.") or name == "actor_logstd") and parameter.requires_grad
        ]
        old_actor_parameters = get_flat_params(actor_parameters)
        old_logprobs = b_logprobs[actor_indices].detach()
        actor_obs = b_obs[actor_indices]
        actor_states = b_states[actor_indices]
        actor_actions = b_actions[actor_indices]
        actor_advantages = b_advantages[actor_indices]
        actor_cost_advantages = b_advantages_c[actor_indices, active_cost]

        # Freeze running observation statistics during repeated Fisher-vector
        # products. They have already been updated by rollout collection.
        self.agent.set_mode('eval')

        def actor_surrogates():
            _, mu, new_logprobs, entropy, _, _ = self.agent.get_action_and_value(
                actor_obs, actor_states, actor_actions
            )
            logratio = new_logprobs - old_logprobs
            ratio = logratio.exp()
            reward_objective = (ratio * actor_advantages).mean()
            if not critic_warmup_active(
                self.attempted_update_iter - 1, self.args.warmup_iters
            ):
                reward_objective = reward_objective + self.cur_ent * entropy.mean()
                if not self.args.beta:
                    reward_objective = reward_objective - self.args.bounds_loss_coef * bound_loss(mu)
            cost_objective = (
                (ratio * actor_cost_advantages).mean() + constraint_baseline
            )
            approx_kl = ((ratio - 1.0) - logratio).mean()
            return reward_objective, cost_objective, approx_kl, entropy.mean()

        reward_objective, cost_objective, _, entropy_loss = actor_surrogates()
        old_reward_objective = reward_objective.detach()
        old_cost_objective = cost_objective.detach()
        reward_gradient = flat_grad(
            reward_objective, actor_parameters, retain_graph=True
        ).detach()
        cost_gradient = flat_grad(cost_objective, actor_parameters).detach()

        accepted_step = False
        approx_kl = torch.zeros((), device=self.device)
        if (
            not critic_warmup_active(
                self.attempted_update_iter - 1, self.args.warmup_iters
            )
            and reward_gradient.norm() > 1e-10
            and cost_gradient.norm() > 1e-10
        ):
            def fisher_vector_product(vector):
                _, _, local_kl, _ = actor_surrogates()
                kl_gradient = flat_grad(
                    local_kl,
                    actor_parameters,
                    create_graph=True,
                    retain_graph=True,
                )
                product = flat_grad(
                    torch.dot(kl_gradient, vector), actor_parameters
                ).detach()
                return product + self.args.cpo_cg_damping * vector

            inv_fisher_reward = conjugate_gradient(
                fisher_vector_product,
                reward_gradient,
                max_iterations=self.args.cpo_cg_iters,
            )
            inv_fisher_cost = conjugate_gradient(
                fisher_vector_product,
                cost_gradient,
                max_iterations=self.args.cpo_cg_iters,
            )
            search_direction, recovery_step = cpo_search_direction(
                inv_fisher_reward,
                inv_fisher_cost,
                reward_gradient,
                cost_gradient,
                old_cost_objective,
                self.args.cpo_max_kl,
            )

            for backtrack in range(self.args.cpo_backtrack_iters):
                fraction = self.args.cpo_backtrack_coeff ** backtrack
                set_flat_params(
                    actor_parameters,
                    old_actor_parameters + fraction * search_direction,
                )
                with torch.no_grad():
                    new_reward, new_cost, new_kl, _ = actor_surrogates()

                finite = bool(torch.isfinite(new_reward) & torch.isfinite(new_cost) & torch.isfinite(new_kl))
                kl_ok = bool(new_kl <= self.args.cpo_max_kl)
                if recovery_step:
                    cost_ok = bool(new_cost < old_cost_objective)
                    reward_ok = True
                else:
                    cost_ok = bool(new_cost <= 1e-6)
                    reward_ok = bool(
                        old_cost_objective > 0
                        or new_reward >= old_reward_objective - 1e-8
                    )
                if finite and kl_ok and cost_ok and reward_ok:
                    reward_objective = new_reward
                    cost_objective = new_cost
                    approx_kl = new_kl
                    accepted_step = True
                    break

        if not accepted_step:
            set_flat_params(actor_parameters, old_actor_parameters)
            with torch.no_grad():
                reward_objective, cost_objective, approx_kl, entropy_loss = actor_surrogates()

        self.agent.set_mode('train')

        # Critic regression uses the full rollout and the same number of epochs
        # and minibatch size as the PPO-family methods.
        b_inds = np.arange(self.args.batch_size)
        v_loss = torch.zeros((), device=self.device)
        v_loss_c = torch.zeros((), device=self.device)
        for _ in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                mb_inds = b_inds[start:start + self.args.minibatch_size]
                mb_value_valid = value_target_valid[mb_inds]
                newvalue, newvalue_c = self.agent.get_value(b_states[mb_inds])
                newvalue = newvalue.view(-1)
                newvalue_c = newvalue_c.view(-1, self.args.num_cost)

                if self.args.clip_vloss:
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss = 0.5 * masked_mean(
                        torch.max(
                            (newvalue - b_returns[mb_inds]) ** 2,
                            (v_clipped - b_returns[mb_inds]) ** 2,
                        ),
                        mb_value_valid,
                    )
                    v_clipped_c = b_values_c[mb_inds] + torch.clamp(
                        newvalue_c - b_values_c[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_c = 0.5 * masked_mean(
                        torch.max(
                            (newvalue_c - b_returns_c[mb_inds]) ** 2,
                            (v_clipped_c - b_returns_c[mb_inds]) ** 2,
                        ),
                        mb_value_valid,
                        dim=0,
                    ).sum()
                else:
                    v_loss = 0.5 * masked_mean(
                        (newvalue - b_returns[mb_inds]) ** 2, mb_value_valid
                    )
                    v_loss_c = 0.5 * masked_mean(
                        (newvalue_c - b_returns_c[mb_inds]) ** 2,
                        mb_value_valid,
                        dim=0,
                    ).sum()

                critic_loss = self.args.vf_coef * (v_loss + v_loss_c)
                self.optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.args.max_grad_norm)
                nn.utils.clip_grad_norm_(self.agent.critic_inst.parameters(), self.args.max_grad_norm)
                nn.utils.clip_grad_norm_(self.agent.critic_t.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

        if self.args.anneal_lr:
            new_lr, _ = self.lr_scheduler.update(self.global_step)
            self.optimizer.param_groups[0]["lr"] = new_lr

        y_pred = b_values.to(torch.float32).cpu().numpy()
        y_true = b_returns.to(torch.float32).cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        self._log_training_metrics(
            -reward_objective,
            v_loss + v_loss_c,
            entropy_loss,
            approx_kl,
            b_advantages[policy_sample_valid],
            explained_var,
            v_loss_c,
            cost_objective,
        )
        return False
