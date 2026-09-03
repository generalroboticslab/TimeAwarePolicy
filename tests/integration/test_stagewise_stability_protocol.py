import unittest

from projects.TimeAwarePolicy.evaluation.launch_stagewise_stability import (
    jobs,
    manifest_path,
    status_path,
)


class StagewiseStabilityProtocolTest(unittest.TestCase):
    def test_full_evaluation_uses_two_tmin_and_fixed_bank_sampling(self):
        job_list = jobs("full")
        self.assertEqual(len(job_list), 6)
        for job in job_list:
            command = job["command"]
            self.assertEqual(job["num_envs"], 2000)
            self.assertNotIn("--goal_time", command)
            self.assertEqual(command[command.index("--goal_speed") + 1], "0.5")
            self.assertEqual(
                command[command.index("--fixed_config_repeats_eval") + 1],
                "2",
            )
            self.assertIn("--fixed_configs_eval", command)
            self.assertIn("--par_configs_eval", command)
            self.assertNotIn("--knn_configs_eval", command)
            self.assertEqual(job["source_bank_size"], 1000)
            self.assertEqual(job["unique_fixed_configs"], 1000)
            self.assertEqual(job["fixed_config_repeats"], 2)

    def test_canary_covers_both_controllers_for_all_tasks(self):
        job_list = jobs("canary")
        self.assertEqual(len(job_list), 6)
        self.assertTrue(all(job["num_envs"] == 128 for job in job_list))
        self.assertTrue(all(job["unique_fixed_configs"] == 64 for job in job_list))
        self.assertEqual(
            {(job["task_key"], job["controller"]) for job in job_list},
            {
                (task, controller)
                for task in ("cube", "gmpour", "cabinet")
                for controller in ("stage_wise", "constant")
            },
        )

    def test_campaign_is_full_stable_stage_only(self):
        self.assertIn("fullstable_repeat2", str(manifest_path("full")))
        self.assertIn("fullstable_repeat2", str(status_path("full")))
        for job in jobs("full"):
            self.assertEqual(
                job["stable_stage_interval"],
                "all executed stable-labelled steps",
            )


if __name__ == "__main__":
    unittest.main()
