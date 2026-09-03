from types import SimpleNamespace

import torch

from core.evaluation.metrics import EvaluationMetricsMixin
from core.training.rollout import RolloutMixin


class ValueAgent:
    def get_value(self, state):
        return torch.tensor([2.0]), None


def rollout_harness(value_bootstrap, timeout):
    harness = RolloutMixin()
    harness.args = SimpleNamespace(
        use_lstm=False,
        use_cost=False,
        num_steps=1,
        value_bootstrap=value_bootstrap,
        gamma=0.9,
        gae_lambda=0.95,
    )
    harness.agent = ValueAgent()
    harness.next_state = torch.zeros(1)
    harness.rewards = torch.tensor([[1.0]])
    harness.values = torch.tensor([[0.5]])
    harness.dones = torch.tensor([[0.0]])
    harness.next_done = torch.tensor([1.0])
    harness.timeouts = torch.tensor([[0.0]])
    harness.next_timeout = torch.tensor([float(timeout)])
    harness.device = torch.device("cpu")
    return harness


def test_rollout_gae_preserves_finite_and_bootstrapped_timeout_semantics():
    finite_returns, finite_advantages, _, _ = rollout_harness(
        value_bootstrap=False,
        timeout=True,
    ).compute_advantages()
    torch.testing.assert_close(finite_advantages, torch.tensor([[0.5]]))
    torch.testing.assert_close(finite_returns, torch.tensor([[1.0]]))

    boot_returns, boot_advantages, _, _ = rollout_harness(
        value_bootstrap=True,
        timeout=True,
    ).compute_advantages()
    torch.testing.assert_close(boot_advantages, torch.tensor([[2.3]]))
    torch.testing.assert_close(boot_returns, torch.tensor([[2.8]]))


def test_evaluation_metrics_keep_success_conditioned_values_separate():
    harness = EvaluationMetricsMixin()
    harness.args = SimpleNamespace(
        paired_stage_eval=False,
        constraint_cost_eval=False,
    )
    harness.step_metrics = {"eps_r": torch.tensor([3.0, 7.0])}
    metric_names = (
        "eps_r",
        "eps_success",
        "eps_time",
        "eps_time_goal",
        "eps_time_p",
        "eps_max_inst",
        "eps_stable_max_inst",
        "eps_lim_inst",
        "eps_sum_inst",
        "interaction_time",
    )
    harness.eps_metrics = {
        name: torch.zeros(4) for name in metric_names
    }
    infos = {
        "success": torch.tensor([1.0, 0.0]),
        "eps_time": torch.tensor([2.0, 99.0]),
        "eps_time_goal": torch.tensor([2.5, 99.0]),
        "eps_time_p": torch.tensor([0.5, 99.0]),
        "eps_max_scevel": torch.tensor([0.2, 99.0]),
        "eps_stable_max_scevel": torch.tensor([0.1, 99.0]),
        "eps_lim_scevel": torch.tensor([0.3, 99.0]),
        "eps_sum_inst": torch.tensor([0.4, 99.0]),
        "interaction_time": torch.tensor([1.5, 99.0]),
    }

    successes = harness.update_episode_metrics(
        torch.tensor([True, True]), infos
    )

    assert successes == 1
    assert harness.eps_metrics["eps_success"][-2:].tolist() == [1.0, 0.0]
    assert harness.eps_metrics["eps_time"][-1].item() == 2.0
    assert 99.0 not in harness.eps_metrics["eps_time"].tolist()
