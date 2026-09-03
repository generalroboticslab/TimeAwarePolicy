import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "projects"
    / "TimeAwarePolicy"
    / "paper"
    / "figures"
    / "training_curves.py"
)
SPEC = importlib.util.spec_from_file_location("plot_training_curves", SCRIPT)
curves = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curves)


class TrainingCurveTest(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.log"
            path.write_text(text)
            return curves.parse_metrics(path)

    def test_parser_accepts_new_checkpoint_score_and_legacy_reward(self):
        records = self.parse(
            "Current Iteration: 1/1500 | Episodes: 10 | "
            "Checkpoint Score: 3.0/4.0 | Success Rate: 0.5/0.6\r"
            "Current Iteration: 2/1500 | Episodes: 20 | "
            "Reward: 5.0/6.0 | Success Rate: 0.7/0.8\r"
        )
        self.assertEqual([record["iteration"] for record in records], [1, 2])
        self.assertEqual(records[0]["checkpoint_score"], 3.0)
        self.assertEqual(records[1]["checkpoint_score"], 5.0)

    def test_schema_two_punctuality_is_already_success_conditioned(self):
        records = [{"iteration": 1, "success_rate": 0.5}]
        curves.merge_wandb(
            records,
            {1: {
                "terminal_time_residual_raw_s": 0.2,
                "episode_metric_schema_version": 2,
            }},
            "cmdp",
        )
        self.assertEqual(records[0]["punctuality_mismatch_success_s"], 0.2)
        self.assertIsNone(records[0]["punctuality_mismatch_raw_s"])

    def test_legacy_punctuality_is_corrected_by_success_rate(self):
        records = [{"iteration": 1, "success_rate": 0.5}]
        curves.merge_wandb(
            records,
            {1: {"terminal_time_residual_raw_s": 0.1}},
            "cmdp",
        )
        self.assertAlmostEqual(
            records[0]["punctuality_mismatch_success_s"], 0.2
        )
        self.assertEqual(records[0]["punctuality_mismatch_raw_s"], 0.1)

    def test_success_conditioned_metrics_are_missing_at_zero_success(self):
        for group, conditioned_key in (
            ("cmdp", "punctuality_mismatch_success_s"),
            ("time_optimal", "remaining_time_success_s"),
        ):
            records = [{"iteration": 1, "success_rate": 0.0}]
            curves.merge_wandb(
                records,
                {1: {
                    "completion_time_success_s": 0.0,
                    "terminal_time_residual_raw_s": 1.5,
                    "episode_metric_schema_version": 2,
                }},
                group,
            )
            self.assertIsNone(records[0]["completion_time_success_s"])
            self.assertIsNone(records[0][conditioned_key])


if __name__ == "__main__":
    unittest.main()
