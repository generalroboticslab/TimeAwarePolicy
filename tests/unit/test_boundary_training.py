import unittest

import torch

from core.agents.utils import (
    apply_boundary_semantics,
    actor_critic_loss,
    compute_stage_end_times,
    compute_staged_time_ratios,
    conjugate_gradient,
    cpo_search_direction,
    critic_warmup_active,
    masked_mean,
    normalize_valid_advantages,
    normalized_time_optimal_terminal_reward,
    paper_time_optimal_terminal_reward,
    repeated_fixed_config_indices,
    rollout_boundary_diagnostics,
    successful_episode_indices,
    update_masked_peak,
    update_lagrange_multipliers_,
)


class BoundaryTrainingTest(unittest.TestCase):
    def setUp(self):
        self.dones = torch.tensor([0.0, 1.0, 0.0, 1.0])
        self.timeouts = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.returns = torch.tensor([5.0, 9.0, 7.0, 11.0])
        self.advantages = torch.tensor([1.0, 0.0, 3.0, 0.0])

    def test_finite_horizon_zeroes_all_boundary_targets(self):
        returns, advantages, policy_valid, value_valid = apply_boundary_semantics(
            self.returns, self.advantages, self.dones, self.timeouts, False
        )

        torch.testing.assert_close(returns, torch.tensor([5.0, 0.0, 7.0, 0.0]))
        torch.testing.assert_close(advantages, self.advantages)
        self.assertEqual(policy_valid.tolist(), [True, False, True, False])
        self.assertEqual(value_valid.tolist(), [True, True, True, True])

    def test_bootstrap_masks_timeout_boundary_from_value_loss(self):
        returns, advantages, policy_valid, value_valid = apply_boundary_semantics(
            self.returns, self.advantages, self.dones, self.timeouts, True
        )

        # A true terminal has a zero target. The timeout boundary retains its
        # rollout value for bookkeeping but is invalid for critic regression.
        torch.testing.assert_close(returns, torch.tensor([5.0, 0.0, 7.0, 11.0]))
        torch.testing.assert_close(advantages, self.advantages)
        self.assertEqual(policy_valid.tolist(), [True, False, True, False])
        self.assertEqual(value_valid.tolist(), [True, True, True, False])

    def test_boundary_action_has_zero_gradient_after_normalization(self):
        raw_advantages = torch.tensor([2.0, 1.0, 0.0])
        policy_valid = torch.tensor([True, True, False])
        advantages, mean, std = normalize_valid_advantages(raw_advantages, policy_valid)

        torch.testing.assert_close(advantages, torch.tensor([1.0, -1.0, 0.0]))
        torch.testing.assert_close(mean, torch.tensor(1.5))
        torch.testing.assert_close(std, torch.tensor(0.5))

        ratio = torch.ones(3, requires_grad=True)
        loss = -(advantages * ratio).mean()
        loss.backward()
        self.assertEqual(float(ratio.grad[-1]), 0.0)

    def test_timeout_boundary_has_zero_critic_gradient(self):
        prediction = torch.tensor([0.0, 0.0], requires_grad=True)
        target = torch.tensor([2.0, 100.0])
        value_valid = torch.tensor([True, False])

        loss = 0.5 * masked_mean((prediction - target) ** 2, value_valid)
        loss.backward()

        self.assertAlmostEqual(float(loss), 2.0)
        torch.testing.assert_close(prediction.grad, torch.tensor([-2.0, 0.0]))

    def test_cost_targets_follow_same_boundary_masks(self):
        returns_c = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        )
        advantages_c = torch.tensor(
            [[2.0, 4.0], [0.0, 0.0], [1.0, 2.0], [0.0, 0.0]]
        )
        returns_c, advantages_c, policy_valid, value_valid = apply_boundary_semantics(
            returns_c, advantages_c, self.dones, self.timeouts, True
        )

        torch.testing.assert_close(
            returns_c,
            torch.tensor([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0], [7.0, 8.0]]),
        )
        normalized, mean, std = normalize_valid_advantages(advantages_c, policy_valid)
        torch.testing.assert_close(normalized[-1], torch.zeros(2))
        torch.testing.assert_close(mean, torch.tensor([1.5, 3.0]))
        torch.testing.assert_close(std, torch.tensor([0.5, 1.0]))
        self.assertEqual(value_valid.tolist(), [True, True, True, False])

    def test_normalized_time_optimal_terminal_reward(self):
        remaining = torch.tensor([-1.0, 0.0, 2.5, 5.0, 7.0])
        reward = normalized_time_optimal_terminal_reward(remaining, 5.0, 1000.0)
        torch.testing.assert_close(
            reward, torch.tensor([0.0, 0.0, 500.0, 1000.0, 1000.0])
        )
        with self.assertRaises(ValueError):
            normalized_time_optimal_terminal_reward(remaining, 0.0, 1000.0)

    def test_paper_time_optimal_terminal_reward_is_not_normalized(self):
        remaining = torch.tensor([-1.0, 0.0, 2.5, 5.0])
        reward = paper_time_optimal_terminal_reward(remaining, 1000.0)
        torch.testing.assert_close(
            reward, torch.tensor([0.0, 0.0, 2500.0, 5000.0])
        )

    def test_rollout_boundary_diagnostics(self):
        diagnostics = rollout_boundary_diagnostics(
            torch.tensor([3.0, 4.0, 5.0, 8.0]),
            self.dones,
            self.timeouts,
        )
        self.assertEqual(diagnostics["boundary_count"], 2)
        self.assertEqual(diagnostics["timeout_count"], 1)
        self.assertEqual(diagnostics["timeout_fraction"], 0.5)
        self.assertEqual(diagnostics["timeout_value_mean"], 8.0)

    def test_first_50_attempted_updates_are_strictly_critic_only(self):
        self.assertTrue(critic_warmup_active(0, 50))
        self.assertTrue(critic_warmup_active(49, 50))
        self.assertFalse(critic_warmup_active(50, 50))

        actor = torch.tensor(2.0, requires_grad=True)
        critic = torch.tensor(3.0, requires_grad=True)
        policy_loss = actor.square()
        value_loss = critic.square()
        entropy = actor
        bounds = actor.abs()
        loss = actor_critic_loss(
            policy_loss,
            value_loss,
            entropy,
            bounds,
            critic_only=True,
            value_coefficient=0.5,
            entropy_coefficient=0.1,
        )
        loss.backward()
        self.assertIsNone(actor.grad)
        self.assertIsNotNone(critic.grad)

    def test_actor_loss_starts_at_update_50(self):
        actor = torch.tensor(2.0, requires_grad=True)
        critic = torch.tensor(3.0, requires_grad=True)
        loss = actor_critic_loss(
            actor.square(),
            critic.square(),
            actor,
            actor.abs(),
            critic_only=critic_warmup_active(50, 50),
            value_coefficient=0.5,
            entropy_coefficient=0.1,
        )
        loss.backward()
        self.assertGreater(abs(float(actor.grad)), 0.0)

    def test_success_conditioned_metrics_exclude_failed_terminals(self):
        terminals = torch.tensor([True, True, False, True])
        success = torch.tensor([1.0, 0.0, 1.0, 1.0])
        values = torch.tensor([0.2, 0.0, 99.0, 0.4])
        indices = successful_episode_indices(terminals, success)
        self.assertEqual(indices.tolist(), [0, 3])
        self.assertAlmostEqual(float(values[indices].mean()), 0.3)

    def test_stable_stage_peak_ignores_efficient_stage_spikes(self):
        peak = torch.zeros(3)
        peak = update_masked_peak(
            peak,
            torch.tensor([0.2, 4.0, 0.5]),
            torch.tensor([True, False, True]),
        )
        torch.testing.assert_close(peak, torch.tensor([0.2, 0.0, 0.5]))
        peak = update_masked_peak(
            peak,
            torch.tensor([0.1, 0.3, 0.8]),
            torch.tensor([True, True, False]),
        )
        torch.testing.assert_close(peak, torch.tensor([0.2, 0.3, 0.5]))

    def test_fixed_bank_is_enumerated_exactly_twice(self):
        indices = repeated_fixed_config_indices(
            torch.arange(2000), num_envs=2000, repeats=2, bank_size=1000
        )
        counts = torch.bincount(indices, minlength=1000)
        torch.testing.assert_close(counts, torch.full((1000,), 2))
        torch.testing.assert_close(indices[:1000], torch.arange(1000))
        torch.testing.assert_close(indices[1000:], torch.arange(1000))

        with self.assertRaises(ValueError):
            repeated_fixed_config_indices(
                torch.arange(2001), num_envs=2001, repeats=2, bank_size=1000
            )

    def test_stagewise_ratios_preserve_half_speed_average(self):
        average = torch.tensor([0.5, 0.5])
        cube = compute_staged_time_ratios(
            average, [0.15, 0.35, 0.15, 0.35], [1, 0, 1, 0], [0.2, 1.0]
        )
        torch.testing.assert_close(
            cube,
            torch.tensor([
                [1.0, 2.0 / 7.0, 1.0, 2.0 / 7.0],
                [1.0, 2.0 / 7.0, 1.0, 2.0 / 7.0],
            ]),
        )
        weighted = (
            cube * torch.tensor([0.15, 0.35, 0.15, 0.35])
        ).sum(dim=1)
        torch.testing.assert_close(weighted, average)

        pour = compute_staged_time_ratios(
            average, [0.5, 0.5], [1, 0], [0.2, 1.0]
        )
        torch.testing.assert_close(
            pour, torch.tensor([[0.8, 0.2], [0.8, 0.2]])
        )

    def test_stage_end_times_cover_every_full_stage(self):
        ends = compute_stage_end_times(
            torch.tensor([2.0, 4.0]), [0.25, 0.75]
        )
        torch.testing.assert_close(
            ends, torch.tensor([[0.5, 2.0], [1.0, 4.0]])
        )

    def test_lagrange_multiplier_is_frozen_during_critic_warmup(self):
        multiplier = torch.tensor([1.0])
        updated = update_lagrange_multipliers_(
            multiplier,
            torch.tensor([2.0]),
            torch.tensor([1.0]),
            0.05,
            100.0,
            critic_only=critic_warmup_active(49, 50),
        )
        self.assertFalse(updated)
        torch.testing.assert_close(multiplier, torch.tensor([1.0]))

        updated = update_lagrange_multipliers_(
            multiplier,
            torch.tensor([2.0]),
            torch.tensor([1.0]),
            0.05,
            100.0,
            critic_only=critic_warmup_active(50, 50),
        )
        self.assertTrue(updated)
        torch.testing.assert_close(multiplier, torch.tensor([1.1]))

    def test_conjugate_gradient_solves_positive_definite_system(self):
        matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
        rhs = torch.tensor([1.0, 2.0])
        solution = conjugate_gradient(lambda vector: matrix @ vector, rhs, max_iterations=10)
        torch.testing.assert_close(matrix @ solution, rhs, atol=1e-5, rtol=1e-5)

    def test_cpo_direction_respects_feasible_linearized_constraint(self):
        # Identity Fisher matrix: the unconstrained reward step would violate
        # x[0] <= 0, so CPO must move along the constraint boundary.
        reward_gradient = torch.tensor([1.0, 1.0])
        cost_gradient = torch.tensor([1.0, 0.0])
        direction, recovery = cpo_search_direction(
            reward_gradient,
            cost_gradient,
            reward_gradient,
            cost_gradient,
            torch.tensor(0.0),
            0.01,
        )
        self.assertFalse(recovery)
        self.assertLessEqual(float(cost_gradient @ direction), 1e-6)
        self.assertLessEqual(float(0.5 * direction @ direction), 0.010001)

    def test_cpo_uses_recovery_step_when_constraint_is_too_infeasible(self):
        reward_gradient = torch.tensor([0.0, 1.0])
        cost_gradient = torch.tensor([1.0, 0.0])
        direction, recovery = cpo_search_direction(
            reward_gradient,
            cost_gradient,
            reward_gradient,
            cost_gradient,
            torch.tensor(1.0),
            0.01,
        )
        self.assertTrue(recovery)
        self.assertLess(float(cost_gradient @ direction), 0.0)
        self.assertLessEqual(float(0.5 * direction @ direction), 0.010001)


if __name__ == "__main__":
    unittest.main()
