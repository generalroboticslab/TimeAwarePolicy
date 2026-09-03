from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from core.agents.agent import Agent


class DummyEnv:
    num_observations = 5
    num_states = 7
    num_actions = 2


def agent_args():
    return SimpleNamespace(
        deterministic=True,
        freeze=False,
        hidden_size=[8, 6, 4],
        norm_obs=False,
        use_cost=True,
        use_fourier=False,
        use_relu=False,
    )


def cloned_state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


class CriticResetTest(unittest.TestCase):
    def test_checkpoint_load_preserves_actor_and_resets_every_critic(self):
        torch.manual_seed(11)
        source = Agent(DummyEnv(), agent_args())
        with torch.no_grad():
            for parameter in source.actor.parameters():
                parameter.fill_(0.25)
            source.actor_logstd.fill_(-0.75)
            for module in source.critic_modules().values():
                for parameter in module.parameters():
                    parameter.fill_(1.5)

        torch.manual_seed(29)
        target = Agent(DummyEnv(), agent_args())
        initial_critics = {
            name: cloned_state(module)
            for name, module in target.critic_modules().items()
        }

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "policy.pt"
            torch.save(source.state_dict(), checkpoint)
            target.load_checkpoint(
                checkpoint,
                map_location=target.device,
                reset_critic=True,
            )

        for name, value in source.actor.state_dict().items():
            torch.testing.assert_close(target.actor.state_dict()[name], value)
        torch.testing.assert_close(target.actor_logstd, source.actor_logstd)
        for critic_name, module in target.critic_modules().items():
            for name, value in initial_critics[critic_name].items():
                torch.testing.assert_close(module.state_dict()[name], value)


if __name__ == "__main__":
    unittest.main()
