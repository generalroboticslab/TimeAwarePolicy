"""Paired stable-stage metric recording for stagewise-control evaluation."""

from pathlib import Path

from core.common.io import save_json


class StagewiseStabilityEvaluationMixin:
    """Record and persist exactly one paired episode per environment."""

    def initialize_paired_stage_metrics(self):
        self.paired_stage_records = []
        self._paired_stage_env_ids = set()

    def _record_paired_stage_metrics(self, terminal_ids, infos):
        """Record each strict-evaluation environment's first episode."""
        init_configs = infos.get("init_configs", {})
        config_rows = infos["paired_config_row"][terminal_ids]
        for env_id_tensor, row_tensor in zip(terminal_ids, config_rows):
            env_id = int(env_id_tensor.item())
            if env_id in self._paired_stage_env_ids:
                continue
            row = int(row_tensor.item())
            if row < 0:
                raise RuntimeError(
                    f"Environment {env_id} has no paired initial-configuration row"
                )
            initial_configuration = {}
            for key, values in init_configs.items():
                if key in {
                    "time_used", "max_linvel", "stable_max_linvel", "sum_linvel"
                }:
                    continue
                if row < len(values):
                    initial_configuration[key] = values[row]
            stable_steps = int(infos["eps_stable_stage_steps"][env_id].item())
            stable_sum = float(infos["eps_sum_inst"][env_id].item())
            self.paired_stage_records.append({
                "env_id": env_id,
                "config_row": row,
                "source_config_index": int(
                    infos["source_config_index"][env_id].item()
                ),
                "success": int(infos["success"][env_id].item()),
                "completion_time_s": float(infos["eps_time"][env_id].item()),
                "stable_object_motion_peak": float(
                    infos["eps_stable_max_scevel"][env_id].item()
                ),
                "stable_object_motion_sum": stable_sum,
                "stable_object_motion_mean": (
                    stable_sum / stable_steps if stable_steps > 0 else 0.0
                ),
                "full_episode_object_motion_peak": float(
                    infos["eps_max_scevel"][env_id].item()
                ),
                "stable_stage_steps": stable_steps,
                "reference_minimum_time_s": float(
                    infos["eps_time_reference"][env_id].item()
                ),
                "scheduled_goal_time_s": float(
                    infos["eps_time_goal"][env_id].item()
                ),
                "stage_time_ratios": infos["stage_time_ratios"][
                    env_id
                ].cpu().tolist(),
                "stage_end_times_s": infos["stage_end_times"][
                    env_id
                ].cpu().tolist(),
                "initial_configuration": initial_configuration,
            })
            self._paired_stage_env_ids.add(env_id)

    def save_paired_stage_metrics(self):
        """Validate and persist the complete paired stable-stage dataset."""
        if len(self.paired_stage_records) != self.args.num_envs:
            raise RuntimeError(
                "Paired stage evaluation recorded "
                f"{len(self.paired_stage_records)} episodes for "
                f"{self.args.num_envs} environments"
            )
        payload = {
            "metrics": {
                "stable_object_motion_mean": (
                    "per-episode time average of the instantaneous "
                    "object-motion proxy over all executed steps labelled stable"
                ),
                "stable_object_motion_peak": (
                    "per-episode maximum instantaneous object-motion proxy "
                    "over all executed steps labelled stable"
                ),
            },
            "controller": (
                "constant_time_ratio"
                if self.args.use_avg_speed
                else "stage_wise_time_ratio"
            ),
            "checkpoint": self.args.checkpoint,
            "index_episode": self.args.index_episode,
            "seed": self.args.seed,
            "budget_portion": self.args.budget_portion,
            "speed_describe": self.args.speed_describe,
            "goal_time": self.args.goal_time,
            "goal_speed": self.args.goal_speed,
            "fixed_config_repeats_eval": self.args.fixed_config_repeats_eval,
            "records": sorted(
                self.paired_stage_records,
                key=lambda item: item["env_id"],
            ),
        }
        save_json(
            payload,
            str(Path(self.args.trajectory_dir) / "paired_stage_metrics.json"),
        )
