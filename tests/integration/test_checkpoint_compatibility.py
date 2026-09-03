import json
from pathlib import Path
from types import SimpleNamespace

import torch

from core.agents.agent import get_agent
from projects.TimeAwarePolicy.arguments.evaluation import get_args


ROOT = Path(__file__).resolve().parents[2]

TIME_AWARE_CHECKPOINTS = (
    ("FrankaCubeStack", "20250717_162724_tw_FrankaCubeStack"),
    ("FrankaGmPour", "20250715_123940_tw_FrankaGmPour"),
    ("FrankaCabinet", "20250730_151924_tw_FrankaCabinet"),
)


def test_included_best_reward_checkpoints_load_strictly_on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint_paths = sorted(
        ROOT.glob("train_res/*/*/checkpoints/eps_best_rew")
    )
    assert len(checkpoint_paths) == 6

    for checkpoint_path in checkpoint_paths:
        state = torch.load(checkpoint_path, map_location="cpu")
        actor_input = state["actor.mlp.0.weight"].shape[1]
        critic_input = state["critic.mlp.0.weight"].shape[1]
        action_count = state["actor.mlp.9.weight"].shape[0] // 2

        config_path = checkpoint_path.parents[1] / "config.json"
        with config_path.open() as config_file:
            arguments = SimpleNamespace(**json.load(config_file))
        environment = SimpleNamespace(
            num_observations=actor_input,
            num_states=critic_input,
            num_actions=action_count,
        )

        agent = get_agent(environment, arguments, device="cpu")
        agent.load_state_dict(state, strict=True)


def test_included_time_aware_checkpoints_resolve_their_calibration_banks():
    for task, checkpoint in TIME_AWARE_CHECKPOINTS:
        arguments = get_args([
            "--task_name", task,
            "--train_res_dir", str(ROOT / "train_res"),
            "--eval_res_dir", str(ROOT / "eval_res"),
            "--checkpoint", checkpoint,
            "--index_episode", "best_rew",
            "--par_configs_eval", "true",
            "--goal_speed", "0.6",
            "--num_envs", "1",
        ])
        bank = (
            ROOT
            / "eval_res"
            / task
            / f"{arguments.par_checkpoint}_EVAL_{arguments.par_index_episode}"
            / "trajectories"
            / "init_configs.json"
        )
        assert arguments.fixed_configs is True
        assert arguments.par_configs is True
        assert not hasattr(arguments, "cabinet_contact_force_limit_n")
        assert bank.is_file()
