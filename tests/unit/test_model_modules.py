from types import SimpleNamespace

import torch

from core.agents.agent import Agent
from core.agents.networks import MLP
from core.agents.normalization import NormalizeReward


def test_split_mlp_keeps_historical_state_dict_layout():
    torch.manual_seed(91)
    model = MLP(5, [4, 3], 2, use_relu=False)
    assert list(model.state_dict()) == [
        "mlp.0.weight", "mlp.0.bias", "mlp.1.weight", "mlp.1.bias",
        "mlp.3.weight", "mlp.3.bias", "mlp.4.weight", "mlp.4.bias",
        "mlp.6.weight", "mlp.6.bias",
    ]


def test_reward_normalizer_honors_cpu_device():
    normalizer = NormalizeReward(4, device="cpu")
    normalized = normalizer.normalize(torch.ones(4), torch.zeros(4))
    assert normalized.shape == (4,)
    assert normalized.device.type == "cpu"


def test_agent_critic_names_remain_checkpoint_compatible(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    environment = SimpleNamespace(
        num_observations=6,
        num_states=7,
        num_actions=2,
    )
    arguments = SimpleNamespace(
        use_cost=True,
        use_relu=False,
        deterministic=True,
        hidden_size=[8, 4],
        use_fourier=False,
        norm_obs=False,
    )
    agent = Agent(environment, arguments)
    keys = set(agent.state_dict())
    assert any(key.startswith("actor.") for key in keys)
    assert any(key.startswith("critic.") for key in keys)
    assert any(key.startswith("critic_t.") for key in keys)
    assert any(key.startswith("critic_inst.") for key in keys)
