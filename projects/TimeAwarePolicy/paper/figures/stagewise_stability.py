"""Build and validate paired stagewise-stability metric candidates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from projects.TimeAwarePolicy.paper.config import require_mapping
from projects.TimeAwarePolicy.paper.style import (
    BarEdgeWidth,
    BaselineLineWidth,
    FillBlueColor,
    FillVioletColor,
    LegendSize,
    NeutralEdgeColor,
    TimeawareColor,
    TimeOptimalColor,
    style_axis,
)


CONTROLLER_DISPLAY = {
    "stage_wise": "Staged tr",
    "constant": "Constant tr",
}
CONTROLLER_COLORS = {
    "stage_wise": TimeawareColor,
    "constant": TimeOptimalColor,
}
CONTROLLER_FILL_COLORS = {
    "stage_wise": FillBlueColor,
    "constant": FillVioletColor,
}
METRICS = ("stable_object_motion_mean", "stable_object_motion_peak")


def _flatten_numeric(value: object) -> np.ndarray:
    if isinstance(value, dict):
        parts = [_flatten_numeric(value[key]) for key in sorted(value)]
        return np.concatenate(parts) if parts else np.empty(0)
    if isinstance(value, list):
        parts = [_flatten_numeric(item) for item in value]
        return np.concatenate(parts) if parts else np.empty(0)
    if isinstance(value, (int, float, bool)):
        return np.asarray([float(value)])
    return np.empty(0)


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    figure.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(figure)


def load_records(
    status_path: Path,
    output_dir: Path,
    evaluation_profile: dict,
) -> tuple[dict[str, dict[str, pd.DataFrame]], list[dict]]:
    """Load and validate the exact fixed-bank paired evaluation."""
    if not status_path.is_file():
        raise FileNotFoundError(status_path)
    tasks = require_mapping(
        evaluation_profile, "tasks", "stagewise_stability_profile"
    )
    source_bank_size = int(evaluation_profile["source_bank_size"])
    repeats = int(evaluation_profile["fixed_config_repeats"])
    expected_records = source_bank_size * repeats
    status = json.loads(status_path.read_text())
    if len(status["jobs"]) != 2 * len(tasks):
        raise RuntimeError("Stagewise stability requires two controllers per task")
    unfinished = [
        job["id"] for job in status["jobs"] if job["status"] != "completed"
    ]
    if unfinished:
        raise RuntimeError(f"Stagewise-stability evaluations unfinished: {unfinished}")

    grouped = {task: {} for task in tasks}
    provenance = []
    raw_rows = []
    configs = {}
    for job in status["jobs"]:
        task_key = job["task_key"]
        if task_key not in tasks:
            raise RuntimeError(f"Unexpected stagewise-stability task: {task_key}")
        task_profile = tasks[task_key]
        command = job["command"]
        required_flags = {
            "--fixed_configs_eval",
            "--par_configs_eval",
            "--fixed_config_repeats_eval",
            "--goal_speed",
        }
        if (
            not required_flags.issubset(command)
            or "--knn_configs_eval" in command
            or "--update_configs" in command
            or "--goal_time" in command
        ):
            raise RuntimeError(f"{job['id']} does not use the fixed-bank protocol")
        output = Path(job["output"])
        paired_path = output / "trajectories" / "paired_stage_metrics.json"
        if not paired_path.is_file():
            raise FileNotFoundError(paired_path)
        payload = json.loads(paired_path.read_text())
        if not np.isclose(payload.get("goal_speed"), task_profile["goal_speed"]):
            raise RuntimeError(f"{job['id']} has the wrong goal speed")
        if payload.get("fixed_config_repeats_eval") != repeats:
            raise RuntimeError(f"{job['id']} has the wrong bank repeat count")
        records = payload["records"]
        if len(records) != expected_records:
            raise RuntimeError(
                f"{job['id']} has {len(records)} records, expected {expected_records}"
            )
        source_indices = [int(record["source_config_index"]) for record in records]
        counts = pd.Series(source_indices).value_counts().sort_index()
        if not np.array_equal(
            counts.index.to_numpy(), np.arange(source_bank_size)
        ) or not (counts.to_numpy() == repeats).all():
            raise RuntimeError(
                f"{job['id']} does not enumerate the complete fixed bank"
            )

        rows = []
        config_list = []
        for record in records:
            initial = record["initial_configuration"]
            stable_steps = int(record["stable_stage_steps"])
            stable_sum = float(record["stable_object_motion_sum"])
            stable_mean = float(record["stable_object_motion_mean"])
            expected_mean = stable_sum / stable_steps if stable_steps > 0 else 0.0
            if not np.isclose(stable_mean, expected_mean, atol=1e-8, rtol=1e-6):
                raise RuntimeError(f"{job['id']} has an invalid time average")
            row = {
                "task_profile": task_key,
                "controller": job["controller"],
                "env_id": int(record["env_id"]),
                "source_config_index": int(record["source_config_index"]),
                "success": int(record["success"]),
                "stable_object_motion_mean": stable_mean,
                "stable_object_motion_peak": float(
                    record["stable_object_motion_peak"]
                ),
                "stable_object_motion_sum": stable_sum,
                "stable_stage_steps": stable_steps,
                "reference_minimum_time_s": float(
                    record["reference_minimum_time_s"]
                ),
                "scheduled_goal_time_s": float(record["scheduled_goal_time_s"]),
                "completion_time_s": float(record["completion_time_s"]),
                "manipulation_distance_m": (
                    float(initial["full_dist"]) if task_key == "cube" else np.nan
                ),
                "evaluation_output": str(output),
            }
            rows.append(row)
            raw_rows.append(row)
            config_list.append(initial)
            goal = row["scheduled_goal_time_s"]
            reference = row["reference_minimum_time_s"]
            if not np.isclose(goal, 2.0 * reference, atol=1e-5, rtol=1e-5):
                raise RuntimeError(f"{job['id']} violates T_goal=2*T_min")
            expected_ratios = (
                [float(task_profile["goal_speed"])]
                * len(task_profile["budget_portion"])
                if job["controller"] == "constant"
                else task_profile["expected_stage_ratios"]
            )
            if not np.allclose(
                record["stage_time_ratios"], expected_ratios, atol=1e-6
            ):
                raise RuntimeError(f"{job['id']} has an invalid stage schedule")
            expected_ends = goal * np.cumsum(task_profile["budget_portion"])
            if not np.allclose(
                record["stage_end_times_s"], expected_ends, atol=1e-5
            ):
                raise RuntimeError(f"{job['id']} has invalid stage milestones")

        frame = pd.DataFrame(rows).sort_values("env_id").reset_index(drop=True)
        grouped[task_key][job["controller"]] = frame
        configs[(task_key, job["controller"])] = config_list
        provenance.append({
            "task": task_key,
            "controller": job["controller"],
            "checkpoint": payload["checkpoint"],
            "index_episode": payload["index_episode"],
            "seed": payload["seed"],
            "budget_portion": payload["budget_portion"],
            "speed_describe": payload["speed_describe"],
            "goal_speed": payload["goal_speed"],
            "source_bank_size": source_bank_size,
            "fixed_config_repeats": repeats,
            "stable_stage_interval": "all executed stable-labelled steps",
            "output": str(output),
            "records": len(records),
        })

    for task in tasks:
        staged = grouped[task]["stage_wise"]
        constant = grouped[task]["constant"]
        for column in ("env_id", "source_config_index"):
            if not np.array_equal(staged[column], constant[column]):
                raise RuntimeError(f"Unpaired {column} for {task}")
        for left, right in zip(
            configs[(task, "stage_wise")], configs[(task, "constant")]
        ):
            a, b = _flatten_numeric(left), _flatten_numeric(right)
            if a.shape != b.shape or not np.allclose(a, b, atol=1e-6, rtol=0):
                raise RuntimeError(f"Initial configurations are not paired for {task}")

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_rows).to_csv(
        data_dir / "paired_fixed_bank_1000x2_per_controller.csv", index=False
    )
    (data_dir / "evaluation_provenance_fullstable.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return grouped, provenance


def _plot_metric(
    summary: pd.DataFrame,
    cube: pd.DataFrame,
    metric: str,
    distance_bins: int,
    output_dir: Path,
    tasks: dict,
) -> pd.DataFrame:
    metric_short = metric[len("stable_object_motion_"):]
    label = (
        "Mean object-motion proxy (m/s)"
        if metric_short == "mean"
        else "Peak object-motion proxy (m/s)"
    )
    edges = np.linspace(
        float(cube.manipulation_distance_m.min()),
        float(cube.manipulation_distance_m.max()),
        distance_bins + 1,
    )
    binned_source = cube.copy()
    binned_source["distance_bin"] = pd.cut(
        binned_source.manipulation_distance_m,
        bins=edges,
        labels=False,
        include_lowest=True,
    )
    binned = binned_source.groupby(
        ["controller", "distance_bin"], observed=True, as_index=False
    ).agg(
        manipulation_distance_m=("manipulation_distance_m", "mean"),
        value=(metric, "mean"),
        std=(metric, "std"),
        samples=(metric, "size"),
    )
    if len(binned) != 2 * distance_bins:
        raise RuntimeError("Stagewise-stability plot contains an empty distance bin")
    binned.to_csv(
        output_dir / "data"
        / f"cube_distance_stable_{metric_short}_binned{distance_bins}.csv",
        index=False,
    )

    figure, axes = plt.subplots(
        1, 2, figsize=(22, 7.2), gridspec_kw={"width_ratios": [0.9, 1.5]}
    )
    task_keys = list(tasks)
    metric_summary = summary[summary.metric == metric]
    x = np.arange(len(task_keys), dtype=float)
    width = 0.32
    for offset, controller, fill, edge in (
        (-width / 2, "stage_wise", FillBlueColor, TimeawareColor),
        (width / 2, "constant", FillVioletColor, TimeOptimalColor),
    ):
        part = metric_summary[
            metric_summary.controller == controller
        ].set_index("task_profile").loc[task_keys]
        axes[0].bar(
            x + offset,
            part.value_mean.to_numpy(float),
            width=width,
            yerr=part.value_std.to_numpy(float),
            capsize=4,
            color=fill,
            edgecolor=edge,
            ecolor=NeutralEdgeColor,
            linewidth=BarEdgeWidth,
            label=CONTROLLER_DISPLAY[controller],
        )
    axes[0].set_xticks(x, [tasks[key]["panel_label"] for key in task_keys])
    axes[0].set_ylabel(label)
    axes[0].legend(loc="upper center", ncol=2, fontsize=LegendSize)
    style_axis(axes[0])

    for controller in ("stage_wise", "constant"):
        part = binned[binned.controller == controller].sort_values(
            "manipulation_distance_m"
        )
        distance = part.manipulation_distance_m.to_numpy(float)
        mean = part.value.to_numpy(float)
        std = part["std"].to_numpy(float)
        axes[1].plot(
            distance,
            mean,
            color=CONTROLLER_COLORS[controller],
            linewidth=BaselineLineWidth,
            alpha=0.95,
            label=CONTROLLER_DISPLAY[controller],
        )
        axes[1].fill_between(
            distance,
            np.maximum(0.0, mean - std),
            mean + std,
            color=CONTROLLER_FILL_COLORS[controller],
            alpha=0.55,
            linewidth=0,
        )
    axes[1].set_xlabel("Manipulation Distance (m)")
    axes[1].set_ylabel(label)
    axes[1].legend(loc="upper center", ncol=2, fontsize=LegendSize)
    style_axis(axes[1])
    if metric_short == "mean":
        axes[0].set_yticks([0.00, 0.06, 0.12])
        axes[1].set_yticks([0.04, 0.06, 0.08])
    figure.tight_layout()
    _save_figure(
        figure,
        output_dir
        / f"stagewise_stable_{metric_short}_object_motion_binned{distance_bins}",
    )
    return binned


def build(
    status_path: Path,
    output_dir: Path,
    evaluation_profile: dict,
    distance_bins: int = 20,
) -> pd.DataFrame:
    """Build both mean and peak stable-stage candidates."""
    tasks = require_mapping(
        evaluation_profile, "tasks", "stagewise_stability_profile"
    )
    grouped, _ = load_records(status_path, output_dir, evaluation_profile)
    source_bank_size = int(evaluation_profile["source_bank_size"])
    repeats = int(evaluation_profile["fixed_config_repeats"])
    summary_rows = []
    cube_rows = []
    for task, task_profile in tasks.items():
        staged = grouped[task]["stage_wise"]
        constant = grouped[task]["constant"]
        both_success = (staged.success == 1) & (constant.success == 1)
        valid_stage = (
            (staged.stable_stage_steps > 0) & (constant.stable_stage_steps > 0)
        )
        analysis_mask = both_success & valid_stage
        for metric in METRICS:
            for controller, frame in (
                ("stage_wise", staged),
                ("constant", constant),
            ):
                values = frame.loc[analysis_mask, metric].to_numpy(float)
                if len(values) < 2:
                    raise RuntimeError(f"Too few valid paired samples for {task}")
                summary_rows.append({
                    "task_profile": task,
                    "task": task_profile["display_name"],
                    "controller": controller,
                    "controller_display": CONTROLLER_DISPLAY[controller],
                    "metric": metric,
                    "paired_rollouts": len(frame),
                    "unique_fixed_configs": source_bank_size,
                    "fixed_config_repeats": repeats,
                    "both_successful_rollouts": int(both_success.sum()),
                    "both_successful_valid_stable_rollouts": int(
                        analysis_mask.sum()
                    ),
                    "controller_success_rate_percent": 100
                    * float(frame.success.mean()),
                    "value_mean": float(values.mean()),
                    "value_std": float(values.std(ddof=1)),
                    "value_se": float(values.std(ddof=1) / np.sqrt(len(values))),
                })
        if task == "cube":
            for controller, frame in (
                ("stage_wise", staged),
                ("constant", constant),
            ):
                selected = frame.loc[analysis_mask].copy()
                selected["controller"] = controller
                cube_rows.append(selected)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "data" / "stable_object_motion_summary.csv", index=False)
    cube = pd.concat(cube_rows, ignore_index=True)
    cube.to_csv(
        output_dir / "data" / "cube_distance_stable_object_motion.csv",
        index=False,
    )
    for metric in METRICS:
        _plot_metric(summary, cube, metric, distance_bins, output_dir, tasks)
    return summary


def validate_outputs(
    summary: pd.DataFrame,
    status_path: Path,
    output_dir: Path,
    distance_bins: int,
    evaluation_profile: dict,
) -> dict:
    """Validate both stability candidates and persist an outcome audit."""
    source_bank_size = int(evaluation_profile["source_bank_size"])
    repeats = int(evaluation_profile["fixed_config_repeats"])
    expected_rollouts = source_bank_size * repeats
    expected = [status_path]
    for metric in ("mean", "peak"):
        expected.extend([
            output_dir
            / f"stagewise_stable_{metric}_object_motion_binned{distance_bins}.pdf",
            output_dir
            / f"stagewise_stable_{metric}_object_motion_binned{distance_bins}.png",
            output_dir / "data"
            / f"cube_distance_stable_{metric}_binned{distance_bins}.csv",
        ])
    expected.extend([
        output_dir / "data" / "stable_object_motion_summary.csv",
        output_dir / "data" / "cube_distance_stable_object_motion.csv",
        output_dir / "data" / "paired_fixed_bank_1000x2_per_controller.csv",
        output_dir / "data" / "evaluation_provenance_fullstable.json",
    ])
    missing = [
        str(path) for path in expected
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Missing stagewise-stability outputs: {missing}")
    if len(summary) != 4 * len(evaluation_profile["tasks"]) or not (
        summary.paired_rollouts == expected_rollouts
    ).all():
        raise RuntimeError("Stagewise summary shape or rollout validation failed")
    if not (summary.unique_fixed_configs == source_bank_size).all() or not (
        summary.fixed_config_repeats == repeats
    ).all():
        raise RuntimeError("Stagewise fixed-bank validation failed")

    outcomes = {}
    for metric in ("mean", "peak"):
        metric_name = f"stable_object_motion_{metric}"
        task = summary[summary.metric == metric_name].pivot(
            index="task_profile", columns="controller", values="value_mean"
        )
        binned = pd.read_csv(
            output_dir / "data"
            / f"cube_distance_stable_{metric}_binned{distance_bins}.csv"
        )
        bins = binned.pivot(
            index="distance_bin", columns="controller", values="value"
        )
        if len(bins) != distance_bins:
            raise RuntimeError(f"Stagewise {metric} plot has the wrong bin count")
        outcomes[metric] = {
            "staged_lower_for_all_tasks": bool(
                (task.stage_wise < task.constant).all()
            ),
            "staged_lower_in_every_distance_bin": bool(
                (bins.stage_wise < bins.constant).all()
            ),
        }
    validation = {
        "status": "passed",
        "figure_style": (
            "original two-controller style; wide two-panel layout; binned "
            "mean curves with one-SD shaded regions"
        ),
        "paired_rollouts_per_controller_and_task": expected_rollouts,
        "source_bank_size": source_bank_size,
        "fixed_config_repeats": repeats,
        "sampling": "exact enumeration of every bank configuration",
        "real_time_goal": "2*T_min per fixed configuration",
        "stable_stage_interval": "all executed stable-labelled steps",
        "distance_bins": distance_bins,
        "both_success_and_valid_stable_stage_filter": True,
        "metric_outcomes": outcomes,
        "checked_files": [str(path) for path in expected],
    }
    (output_dir / "data" / "stagewise_stability_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    return validation
