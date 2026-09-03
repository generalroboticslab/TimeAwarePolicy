"""Episode-level evaluation aggregation and result persistence."""

import os

import numpy as np
import torch

from core.agents.agent import update_tensor_buffer
from core.common.io import save_json, write_csv_line


class EvaluationMetricsMixin:
    def update_episode_metrics(self, terminal_index, infos):
        """Update episode metrics when episodes complete."""
        terminal_ids = terminal_index.nonzero().flatten()
        success_buf = infos["success"][terminal_index]
        success_ids = terminal_ids[success_buf.to(torch.bool)]

        if self.args.paired_stage_eval:
            self._record_paired_stage_metrics(terminal_ids, infos)

        update_tensor_buffer(
            self.eps_metrics["eps_r"],
            self.step_metrics["eps_r"][terminal_index],
        )
        update_tensor_buffer(
            self.eps_metrics["eps_success"],
            infos["success"][terminal_index],
        )

        if len(success_ids) > 0:
            metric_sources = {
                "eps_time": "eps_time",
                "eps_time_goal": "eps_time_goal",
                "eps_time_p": "eps_time_p",
                "eps_max_inst": "eps_max_scevel",
                "eps_stable_max_inst": "eps_stable_max_scevel",
                "eps_lim_inst": "eps_lim_scevel",
                "eps_sum_inst": "eps_sum_inst",
                "interaction_time": "interaction_time",
            }
            for metric, source in metric_sources.items():
                update_tensor_buffer(
                    self.eps_metrics[metric], infos[source][success_ids]
                )
            if self.args.constraint_cost_eval:
                update_tensor_buffer(
                    self.eps_metrics["eps_instability_cost"],
                    self.step_metrics["eps_instability_cost"][success_ids],
                )

        for key in self.step_metrics.keys():
            self.step_metrics[key][terminal_index] = 0.0

        return len(success_ids)

    def compute_average_metrics(self, num_episodes, num_success_eps):
        """Compute average and standard-deviation metrics."""
        for key in self.eps_metrics_avg.keys():
            eps_index = (
                num_episodes
                if key in ["eps_r", "eps_success"]
                else num_success_eps
            )
            values = self.eps_metrics[key][-eps_index:]
            self.eps_metrics_avg[key] = torch.mean(values).item()
            std_value = torch.std(values)
            self.eps_metrics_std[key] = (
                std_value.item() if not torch.isnan(std_value) else 0.0
            )

    def update_speed_time_dict(self, cur_goal_speed, cur_dynamic_v):
        """Append one condition's aggregate metrics."""
        self.speed_and_time_dict["time_ratio"].append(cur_goal_speed)

        dynamic_names = {
            "FrankaCubeStack": "disturbance_v",
            "FrankaGmPour": "num_gms",
            "FrankaCabinet": "friction_mul",
        }
        self.speed_and_time_dict[dynamic_names[self.args.task_name]].append(
            cur_dynamic_v
        )

        metric_names = (
            ("time_used", "eps_time"),
            ("time_goal", "eps_time_goal"),
            ("time_mismatch", "eps_time_p"),
            ("max_inst", "eps_max_inst"),
            ("stable_max_inst", "eps_stable_max_inst"),
            ("thred_inst", "eps_lim_inst"),
            ("sum_inst", "eps_sum_inst"),
            ("interaction_time", "interaction_time"),
        )
        for output_name, metric_name in metric_names:
            self.speed_and_time_dict[output_name].append(
                [
                    self.eps_metrics_avg[metric_name],
                    self.eps_metrics_std[metric_name],
                ]
            )
        if self.args.constraint_cost_eval:
            self.speed_and_time_dict["instability_cost"].append(
                [
                    self.eps_metrics_avg["eps_instability_cost"],
                    self.eps_metrics_std["eps_instability_cost"],
                ]
            )
        self.speed_and_time_dict["success_rate"].append(
            self.eps_metrics_avg["eps_success"]
        )

    def save_results(self, num_episodes, machine_time, num_eps_recorded, infos):
        """Persist one completed evaluation condition."""
        if not self.args.saving:
            return

        csv_result = {
            "target_episodes": self.args.target_episodes,
            "success_rate": self.eps_metrics_avg["eps_success"],
            "avg_reward": self.eps_metrics_avg["eps_r"],
            "avg_sum_eps_inst": self.eps_metrics_avg["eps_sum_inst"],
            "machine_time": machine_time,
        }
        write_csv_line(self.args.csv_file_path, csv_result)
        print(f"Saved evaluation CSV to {self.args.csv_file_path}")

        meta_data = {
            "episode": num_episodes,
            "episode_success": self.eps_metrics["eps_success"][
                -num_episodes:
            ].cpu().tolist(),
            "episode_time": self.eps_metrics["eps_time"][
                -num_episodes:
            ].cpu().tolist(),
            "episode_time_goal": self.eps_metrics["eps_time_goal"][
                -num_episodes:
            ].cpu().tolist(),
            "speed_and_time": self.speed_and_time_dict,
        }
        save_json(
            meta_data,
            os.path.join(self.args.trajectory_dir, "meta_data.json"),
        )

        if self.args.paired_stage_eval:
            self.save_paired_stage_metrics()

        if self.args.record_init_configs and not self.args.paired_stage_eval:
            self._save_init_configs(infos, num_eps_recorded)

    def _save_init_configs(self, infos, num_eps_recorded):
        """Save a time-balanced set of successful initial configurations."""
        filter_configs = {}
        for index, time_used in enumerate(infos["init_configs"]["time_used"]):
            if time_used > 0:
                for key, value in infos["init_configs"].items():
                    filter_configs.setdefault(key, []).append(value[index])

        valid_time_used = filter_configs["time_used"]
        num_valid_configs = len(valid_time_used)
        print(f"Valid Configs Num: {num_valid_configs}")

        if not self.args.update_configs:
            if self.args.strict_eval:
                self.args.target_success_eps = num_valid_configs
                self.args.target_record_eps = num_valid_configs

            print(
                f"Starts to downsample the ({num_valid_configs}) configs to "
                f"({self.args.target_record_eps}) records"
            )

            min_utime, max_utime = np.min(valid_time_used), np.max(valid_time_used)
            utime_bins = np.linspace(min_utime, max_utime, 10)
            config2bin_idxs = np.digitize(valid_time_used, utime_bins, right=True)
            bin_groups = {bin_idx: [] for bin_idx in range(len(utime_bins))}
            for index, bin_idx in enumerate(config2bin_idxs):
                bin_groups[bin_idx].append(index)

            num_records = 0
            bin_pointer = 0
            recorded_configs = {}
            utime_counts = [0] * len(utime_bins)
            while num_records < self.args.target_record_eps:
                if len(bin_groups[bin_pointer]) == 0:
                    bin_pointer += 1
                else:
                    config_idx = bin_groups[bin_pointer].pop()
                    for key, value in filter_configs.items():
                        recorded_configs.setdefault(key, []).append(
                            value[config_idx]
                        )
                    utime_counts[bin_pointer] += 1
                    num_records += 1
                    bin_pointer += 1

                bin_pointer %= len(utime_bins)
                if all(len(group) == 0 for group in bin_groups.values()):
                    raise Exception(
                        "All Bins are empty. There are not enough configs to retrieve"
                    )

            from core.evaluation.visualization import plot_utime_dataset

            plot_utime_dataset(
                utime_bins,
                utime_counts,
                save_dir=self.args.instance_dir,
            )
        else:
            original_configs = self.envs.env_configs
            recorded_configs = filter_configs
            edit_configs = 0
            for index, time_used in enumerate(recorded_configs["time_used"]):
                org_time_used = original_configs["time_used"][index].item()
                org_max_linvel = original_configs["max_linvel"][index].item()
                if time_used > org_time_used:
                    recorded_configs["time_used"][index] = org_time_used
                    recorded_configs["max_linvel"][index] = org_max_linvel
                    edit_configs += 1

            print(
                f"Reverted {edit_configs}/{len(original_configs['time_used'])} "
                "configs to the original time used"
            )

        recorded_configs["avg_time_used"] = np.mean(
            recorded_configs["time_used"]
        )
        save_json(
            recorded_configs,
            os.path.join(self.args.trajectory_dir, "init_configs.json"),
        )
