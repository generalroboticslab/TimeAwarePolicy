import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from core.training.checkpointing import CheckpointingMixin


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "projects"
    / "TimeAwarePolicy"
    / "initializer_quality"
    / "select_checkpoints.py"
)
SPEC = importlib.util.spec_from_file_location("quality_calibration", SCRIPT_PATH)
QUALITY_CALIBRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALITY_CALIBRATION)


class RecordingAgent:
    def __init__(self):
        self.suffixes = []

    def save_checkpoint(self, folder_path, suffix, reward_normalizer):
        self.suffixes.append(suffix)


class CandidateTrainer(CheckpointingMixin):
    pass


def test_candidate_capture_starts_at_ratio_zero_and_uses_five_accepted_updates(tmp_path):
    trainer = CandidateTrainer()
    trainer.args = SimpleNamespace(
        saving=True,
        pre_train=True,
        quality_candidate_interval=5,
        quality_candidate_start_success=0.90,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        trajectory_dir=str(tmp_path),
    )
    trainer.agent = RecordingAgent()
    trainer.reward_normalizer = object()
    trainer.ready_to_record = True
    trainer.curri_ratio = 0
    trainer.cur_success_rate = 0.89
    trainer.global_update_iter = trainer.attempted_update_iter = 100
    trainer.global_step = 1000
    trainer.quality_candidate_start_update = None
    trainer.quality_candidate_last_update = None
    trainer.quality_candidates = {}
    trainer.meta_data = {"quality_candidates": trainer.quality_candidates, "training_info": {}}

    assert trainer.save_quality_candidate() is False
    trainer.cur_success_rate = 0.90
    assert trainer.save_quality_candidate() is True
    assert trainer.agent.suffixes == ["candidate_u00100"]

    trainer.curri_ratio = 0.3
    trainer.global_update_iter = trainer.attempted_update_iter = 104
    assert trainer.save_quality_candidate() is False
    trainer.global_update_iter = trainer.attempted_update_iter = 105
    assert trainer.save_quality_candidate() is True
    assert trainer.agent.suffixes[-1] == "candidate_u00105"
    assert trainer.quality_candidates["candidate_u00105"]["label_status"].startswith("unlabeled")


def test_distinct_assignment_uses_measured_held_out_success():
    evaluations = [
        {"name": "a", "success_rate": 0.39},
        {"name": "b", "success_rate": 0.62},
        {"name": "c", "success_rate": 0.78},
        {"name": "d", "success_rate": 0.94},
        {"name": "e", "success_rate": 0.97},
    ]
    selected = QUALITY_CALIBRATION.assign_distinct(evaluations, [0.4, 0.6, 0.8, 0.95])
    assert [item["name"] for item in selected] == ["a", "b", "c", "d"]


def test_evaluation_output_matches_evaluator_name_limit(tmp_path):
    output = QUALITY_CALIBRATION.evaluation_output(
        tmp_path, "Task", {"final_name": "x" * 249}, "candidate_u00010"
    )
    assert len(output.name) == 250
    assert output.name.endswith("_EVAL_candidate_u00010")


def test_discovery_rejects_missing_candidate_pair(tmp_path):
    producer = tmp_path / "producer"
    (producer / "checkpoints").mkdir(parents=True)
    (producer / "trajectories").mkdir()
    metadata = {
        "quality_candidates": {
            "candidate_u00010": {
                "accepted_update": 10,
                "attempted_update": 10,
                "rolling_success_rate": 0.9,
                "curriculum_ratio": 0,
            }
        }
    }
    (producer / "trajectories" / "meta_data.json").write_text(json.dumps(metadata))
    (producer / "checkpoints" / "eps_candidate_u00010").touch()

    try:
        QUALITY_CALIBRATION.discover_candidates(producer)
    except FileNotFoundError as error:
        assert "reward normalizer" in str(error)
    else:
        raise AssertionError("missing reward normalizer was accepted")


def test_sealed_bank_records_actual_success_and_distinct_sources(tmp_path):
    producer = tmp_path / "producer"
    (producer / "checkpoints").mkdir(parents=True)
    (producer / "trajectories").mkdir()
    (producer / "config.json").write_text(json.dumps({"task_name": "Task"}))
    evaluations = []
    for update, success in zip((10, 15, 20, 25), (0.35, 0.58, 0.82, 0.96)):
        name = f"candidate_u{update:05d}"
        policy = producer / "checkpoints" / f"eps_{name}"
        normalizer = producer / "checkpoints" / f"rew_norm_eps_{name}"
        policy.write_bytes(f"policy-{update}".encode())
        normalizer.write_bytes(f"normalizer-{update}".encode())
        evaluations.append({
            "name": name,
            "accepted_update": update,
            "attempted_update": update,
            "rolling_success_rate": 0.9,
            "curriculum_ratio": update / 100,
            "policy": policy,
            "normalizer": normalizer,
            "success_rate": success,
            "output": tmp_path / f"eval-{update}",
        })

    protocol = {"num_envs": 2000, "seed": 123456}
    bank = tmp_path / "bank"
    QUALITY_CALIBRATION.seal_quality_bank(
        producer, bank, evaluations, [0.4, 0.6, 0.8, 0.95], protocol
    )
    metadata = json.loads((bank / "trajectories" / "meta_data.json").read_text())
    selections = metadata["milestone"]

    assert selections["quality_40"]["success_rate"] == 0.35
    assert selections["quality_95"]["success_rate"] == 0.96
    assert len({record["source_index"] for record in selections.values()}) == 4
    assert all(record["source"].startswith("held_out") for record in selections.values())
