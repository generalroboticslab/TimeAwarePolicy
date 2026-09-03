import json
from types import SimpleNamespace

import pytest
import torch

from projects.TimeAwarePolicy.evaluation.stagewise_stability import (
    StagewiseStabilityEvaluationMixin,
)


class Recorder(StagewiseStabilityEvaluationMixin):
    pass


def test_paired_stage_mixin_records_mean_and_persists_payload(tmp_path):
    recorder = Recorder()
    recorder.args = SimpleNamespace(
        num_envs=1,
        use_avg_speed=False,
        checkpoint="checkpoint",
        index_episode="best_rew",
        seed=123456,
        budget_portion=[0.5, 0.5],
        speed_describe=[1, 0],
        goal_time=None,
        goal_speed=0.5,
        fixed_config_repeats_eval=2,
        trajectory_dir=str(tmp_path),
    )
    recorder.initialize_paired_stage_metrics()
    infos = {
        "init_configs": {
            "time_used": [1.0],
            "full_dist": [0.4],
        },
        "paired_config_row": torch.tensor([0]),
        "source_config_index": torch.tensor([7]),
        "success": torch.tensor([1]),
        "eps_time": torch.tensor([2.0]),
        "eps_stable_max_scevel": torch.tensor([0.3]),
        "eps_sum_inst": torch.tensor([0.4]),
        "eps_stable_stage_steps": torch.tensor([2]),
        "eps_max_scevel": torch.tensor([0.5]),
        "eps_time_reference": torch.tensor([1.0]),
        "eps_time_goal": torch.tensor([2.0]),
        "stage_time_ratios": torch.tensor([[0.8, 0.2]]),
        "stage_end_times": torch.tensor([[1.0, 2.0]]),
    }

    recorder._record_paired_stage_metrics(torch.tensor([0]), infos)
    recorder.save_paired_stage_metrics()

    payload = json.loads((tmp_path / "paired_stage_metrics.json").read_text())
    assert payload["controller"] == "stage_wise_time_ratio"
    assert payload["records"][0]["source_config_index"] == 7
    assert payload["records"][0]["stable_object_motion_mean"] == pytest.approx(0.2)
