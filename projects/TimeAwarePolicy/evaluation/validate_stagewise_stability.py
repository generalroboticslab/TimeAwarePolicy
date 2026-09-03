#!/usr/bin/env python3
"""Validate and summarize paired stagewise-stability evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from projects.TimeAwarePolicy.paper.config import (
    load_profile,
    require_mapping,
    require_sequence,
    sha256_file,
)


DEFAULT_PROFILE = (
    ROOT
    / "projects"
    / "TimeAwarePolicy"
    / "paper"
    / "configs"
    / "stagewise_stability_evaluation.json"
)


def flatten_numeric(value: object) -> np.ndarray:
    if isinstance(value, dict):
        parts = [flatten_numeric(value[key]) for key in sorted(value)]
        return np.concatenate(parts) if parts else np.empty(0)
    if isinstance(value, list):
        parts = [flatten_numeric(item) for item in value]
        return np.concatenate(parts) if parts else np.empty(0)
    if isinstance(value, (int, float, bool)):
        return np.asarray([float(value)])
    return np.empty(0)


def load_and_validate(
    status_path: Path, profile_path: Path = DEFAULT_PROFILE
) -> dict:
    profile_path = profile_path.resolve()
    profile = load_profile(profile_path, "stagewise_stability_evaluation")
    tasks = require_mapping(profile, "tasks", "profile")
    repeats = int(profile["fixed_config_repeats"])
    source_bank_size = int(profile["source_bank_size"])
    status = json.loads(status_path.read_text())
    if len(status["jobs"]) != 2 * len(tasks):
        raise RuntimeError("wrong number of stagewise-stability jobs")
    grouped = {}
    schedule_validation = {}
    for job in status["jobs"]:
        if job["status"] != "completed":
            raise RuntimeError(f"unfinished job: {job['id']}")
        artifact = (
            Path(job["output"])
            / "trajectories"
            / "paired_stage_metrics.json"
        )
        payload = json.loads(artifact.read_text())
        records = sorted(payload["records"], key=lambda item: item["env_id"])
        if len(records) != job["num_envs"]:
            raise RuntimeError(f"wrong record count for {job['id']}")
        task = job["task_key"]
        task_spec = require_mapping(tasks, task, "tasks")
        goal_speed = float(task_spec["goal_speed"])
        budget_portions = np.asarray(
            require_sequence(task_spec, "budget_portion", task)
        )
        staged_ratios = list(
            require_sequence(task_spec, "expected_stage_ratios", task)
        )
        if not np.isclose(payload["goal_speed"], goal_speed):
            raise RuntimeError(f"wrong goal speed for {job['id']}")
        if payload.get("fixed_config_repeats_eval") != repeats:
            raise RuntimeError(f"wrong fixed-config repeat count for {job['id']}")

        source_indices = [row["source_config_index"] for row in records]
        source_counts = {
            index: source_indices.count(index) for index in set(source_indices)
        }
        expected_unique = source_bank_size
        if sorted(source_counts) != list(range(expected_unique)) or any(
            count != repeats for count in source_counts.values()
        ):
            raise RuntimeError(
                f"{job['id']} does not enumerate each fixed configuration twice"
            )

        expected_ratios = (
            [goal_speed] * len(budget_portions)
            if job["controller"] == "constant"
            else staged_ratios
        )
        maximum_goal_error = 0.0
        for record in records:
            reference = float(record["reference_minimum_time_s"])
            goal = float(record["scheduled_goal_time_s"])
            maximum_goal_error = max(maximum_goal_error, abs(goal - 2 * reference))
            if not np.allclose(
                record["stage_time_ratios"], expected_ratios, atol=1e-6, rtol=0
            ):
                raise RuntimeError(f"wrong stage ratios in {job['id']}")
            ends = goal * np.cumsum(budget_portions)
            if not np.allclose(
                record["stage_end_times_s"], ends, atol=1e-5, rtol=1e-5
            ):
                raise RuntimeError(f"wrong stage windows in {job['id']}")
            stable_steps = int(record["stable_stage_steps"])
            expected_mean = (
                float(record["stable_object_motion_sum"]) / stable_steps
                if stable_steps > 0 else 0.0
            )
            if not np.isclose(
                record["stable_object_motion_mean"], expected_mean,
                atol=1e-8, rtol=1e-6,
            ):
                raise RuntimeError(f"wrong stable-stage mean in {job['id']}")
        if maximum_goal_error > 1e-5:
            raise RuntimeError(f"T_goal != 2*T_min in {job['id']}")
        grouped[(task, job["controller"])] = records
        schedule_validation[job["id"]] = {
            "records": len(records),
            "unique_fixed_configs": len(source_counts),
            "fixed_config_repeats": repeats,
            "expected_stage_ratios": expected_ratios,
            "maximum_abs_Tgoal_minus_2Tmin_s": maximum_goal_error,
            "minimum_stable_steps": min(
                record["stable_stage_steps"] for record in records
            ),
        }

    summaries = {}
    for task in tasks:
        staged = grouped[(task, "stage_wise")]
        constant = grouped[(task, "constant")]
        for left, right in zip(staged, constant):
            if left["env_id"] != right["env_id"]:
                raise RuntimeError(f"unpaired environment IDs for {task}")
            if left["source_config_index"] != right["source_config_index"]:
                raise RuntimeError(f"unpaired source configuration IDs for {task}")
            left_config = flatten_numeric(left["initial_configuration"])
            right_config = flatten_numeric(right["initial_configuration"])
            if left_config.shape != right_config.shape or not np.allclose(
                left_config, right_config, atol=1e-6, rtol=0
            ):
                raise RuntimeError(f"unpaired initial configurations for {task}")

        valid = np.asarray([
            bool(left["success"])
            and bool(right["success"])
            and left["stable_stage_steps"] > 0
            and right["stable_stage_steps"] > 0
            for left, right in zip(staged, constant)
        ])
        task_summary = {
            "paired_rollouts": len(staged),
            "paired_both_success_valid_stable_stage": int(valid.sum()),
            "stage_wise_success_rate": float(
                np.mean([row["success"] for row in staged])
            ),
            "constant_success_rate": float(
                np.mean([row["success"] for row in constant])
            ),
        }
        for metric in (
            "stable_object_motion_mean", "stable_object_motion_peak"
        ):
            staged_values = np.asarray([row[metric] for row in staged])[valid]
            constant_values = np.asarray([row[metric] for row in constant])[valid]
            differences = staged_values - constant_values
            difference_se = differences.std(ddof=1) / np.sqrt(len(differences))
            task_summary[metric] = {
                "stage_wise_mean": float(staged_values.mean()),
                "stage_wise_std": float(staged_values.std(ddof=1)),
                "constant_mean": float(constant_values.mean()),
                "constant_std": float(constant_values.std(ddof=1)),
                "internal_staged_minus_constant_mean": float(differences.mean()),
                "internal_staged_minus_constant_std": float(
                    differences.std(ddof=1)
                ),
                "internal_staged_minus_constant_normal_95pct_ci": [
                    float(differences.mean() - 1.96 * difference_se),
                    float(differences.mean() + 1.96 * difference_se),
                ],
                "internal_staged_lower_fraction": float((differences < 0).mean()),
            }
        summaries[task] = task_summary
    return {
        "status": "validated",
        "source_status": str(status_path.resolve()),
        "stable_stage_interval": "all executed stable-labelled steps",
        "source_bank_size": source_bank_size,
        "fixed_config_repeats": repeats,
        "evaluation_profile": str(profile_path),
        "evaluation_profile_sha256": sha256_file(profile_path),
        "schedule_validation": schedule_validation,
        "internal_diagnostics": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("status", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    args = parser.parse_args()
    result = load_and_validate(args.status, args.profile)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
