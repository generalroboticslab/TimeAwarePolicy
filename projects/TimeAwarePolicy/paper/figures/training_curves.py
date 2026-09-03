#!/usr/bin/env python3
"""Plot every available training log in a three-task campaign.

This is intentionally read-only with respect to training outputs.  It takes a
snapshot of ``status.json`` and writes plots plus the parsed numeric data below
the campaign's ``results`` directory.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from wandb.proto import wandb_internal_pb2  # noqa: E402
from wandb.sdk.internal.datastore import DataStore  # noqa: E402


NUMBER = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan)"
ITERATION_RE = re.compile(r"Current Iteration: (\d+)/(\d+) \| Episodes: (\d+)")
SCORE_RE = re.compile(rf"(?:Checkpoint Score|Reward): ({NUMBER})/({NUMBER})")
SUCCESS_RE = re.compile(rf"Success Rate: ({NUMBER})/({NUMBER})")
COST_RE = re.compile(rf"Cost: ({NUMBER})")
EPISODE_TIME_RE = re.compile(rf"Max Episode Time: ({NUMBER})/({NUMBER})")

METRICS = (
    ("success_rate", "Rolling success (%)", 100.0),
    ("checkpoint_score", "Rolling checkpoint score", 1.0),
    ("cost", "Rolling instability cost", 1.0),
    ("completion_time_success_s", "Successful episode completion time (s)", 1.0),
    ("punctuality_mismatch_success_s", "Punctuality mismatch, successful episodes (s)", 1.0),
    ("punctuality_mismatch_raw_s", "Unconditioned punctuality log (s)", 1.0),
    ("remaining_time_success_s", "Remaining time at successful completion (s)", 1.0),
    ("curriculum_ratio", "Curriculum ratio", 1.0),
    ("lagrange_multiplier", "PPO-Lagrangian multiplier", 1.0),
    ("max_episode_time", "Maximum episode time (s)", 1.0),
)


def json_number(value):
    try:
        parsed = json.loads(value)
        return float(parsed) if parsed is not None else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def load_wandb_histories(campaign_dir: Path):
    """Return offline W&B histories indexed by display name.

    The console intentionally prints only a compact subset.  Episode timing,
    curriculum ratio, and multiplier trajectories are persisted in the binary
    offline W&B stream, so those fields are merged into the console records.
    """
    histories = {}
    wandb_root = campaign_dir / "wandb" / "wandb"
    for path in sorted(wandb_root.glob("offline-run-*/run-*.wandb")):
        store = DataStore()
        store.open_for_scan(str(path))
        record = wandb_internal_pb2.Record()
        display_name = None
        rows = []
        while True:
            try:
                data = store.scan_data()
            except AssertionError:
                # A live offline run can end in a partially flushed record.
                # All complete records scanned before it remain usable.
                break
            if data is None:
                break
            record.ParseFromString(data)
            if record.HasField("run"):
                display_name = record.run.display_name or record.run.name
            if record.HasField("history"):
                rows.append({item.key: item.value_json for item in record.history.item})
        if not display_name:
            continue
        by_iteration = {}
        last_iteration = None
        for row in rows:
            iteration = json_number(row.get("misc/attempted_updates"))
            if iteration is None:
                iteration = json_number(row.get("iterations"))
            if iteration is None:
                iteration = json_number(row.get("misc/global_iterations"))
            if iteration is not None:
                last_iteration = int(iteration)
            if last_iteration is None:
                continue
            merged = by_iteration.setdefault(last_iteration, {})
            mappings = {
                "reward/eps_time": "completion_time_success_s",
                "reward/eps_time_p": "terminal_time_residual_raw_s",
                "reward/curriculum_ratio": "curriculum_ratio",
                "reward/rolling_checkpoint_score": "checkpoint_score",
                "misc/episode_metric_schema_version": "episode_metric_schema_version",
                "train/lagrange_multiplier_1": "lagrange_multiplier",
            }
            for source, target in mappings.items():
                value = json_number(row.get(source))
                if value is not None:
                    merged[target] = value
        histories[display_name] = by_iteration
    return histories


def matching_wandb_history(job: dict, histories: dict):
    result = job.get("result_directory")
    if result and Path(result).name in histories:
        return histories[Path(result).name]
    prefix = job.get("force_name")
    matches = [value for name, value in histories.items() if prefix and name.startswith(prefix)]
    return matches[-1] if len(matches) == 1 else {}


def merge_wandb(records: list[dict], history: dict, result_group: str):
    for record in records:
        extra = history.get(record["iteration"], {})
        record.update(extra)
        raw = record.get("terminal_time_residual_raw_s")
        success = record.get("success_rate")
        if success is None or success <= 0:
            # A success-conditioned statistic is undefined when its matching
            # rolling window contains no successes. W&B may otherwise retain
            # a zero or stale value from the successful-episode accumulator.
            record["completion_time_success_s"] = None
            corrected = None
            success_conditioned = (
                record.get("episode_metric_schema_version", 1) >= 2
            )
        else:
            # Schema 2 records eps_time_p over successful episodes directly.
            # Schema 1 mixed failures in as zero and requires division by the
            # success rate.
            success_conditioned = (
                record.get("episode_metric_schema_version", 1) >= 2
            )
            if success_conditioned:
                corrected = raw
            else:
                corrected = raw / success if raw is not None else None
        if result_group == "cmdp":
            record["punctuality_mismatch_raw_s"] = (
                None if success_conditioned else raw
            )
            record["punctuality_mismatch_success_s"] = corrected
        else:
            record["remaining_time_success_s"] = corrected


def parse_metrics(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(errors="replace").replace("\r", "\n").splitlines():
        iteration = ITERATION_RE.search(line)
        score = SCORE_RE.search(line)
        success = SUCCESS_RE.search(line)
        if not (iteration and score and success):
            continue
        cost = COST_RE.search(line)
        episode_time = EPISODE_TIME_RE.search(line)
        records.append({
            "iteration": int(iteration.group(1)),
            "total_iterations": int(iteration.group(2)),
            "episodes": int(iteration.group(3)),
            "checkpoint_score": float(score.group(1)),
            "best_checkpoint_score": float(score.group(2)),
            "success_rate": float(success.group(1)),
            "best_success_rate": float(success.group(2)),
            "cost": float(cost.group(1)) if cost else None,
            "max_episode_time": float(episode_time.group(1)) if episode_time else None,
            "best_max_episode_time": (
                float(episode_time.group(2)) if episode_time else None
            ),
        })
    # A log can contain a repeated line at a tool-capture boundary.  The last
    # occurrence is authoritative.
    by_iteration = {record["iteration"]: record for record in records}
    return [by_iteration[key] for key in sorted(by_iteration)]


def rolling(values: list[float], window: int) -> list[float]:
    output = []
    finite_window = []
    for value in values:
        finite_window.append(value)
        if len(finite_window) > window:
            finite_window.pop(0)
        valid = [item for item in finite_window if math.isfinite(item)]
        output.append(mean(valid) if valid else math.nan)
    return output


def metric_points(records: list[dict], key: str, scale: float = 1.0):
    points = [
        (record["iteration"], record[key] * scale)
        for record in records
        if record.get(key) is not None and math.isfinite(record[key])
    ]
    return [item[0] for item in points], [item[1] for item in points]


def comparison(job: dict) -> str:
    if (
        job.get("quality") is not None
        and job.get("result_group") in {"initial_quality", "time_optimal"}
    ):
        return f"q{job['quality']}"
    if job.get("result_group") == "cmdp":
        return job["method"]
    return "ordinary PPO producer"


def display_comparison(value: str) -> str:
    names = {
        "np3o": "N-P3O",
        "ppo_lagrangian": "PPO-Lagrangian",
        "cpo": "CPO",
    }
    return names.get(value, value.upper() if value.startswith("q") else value)


def available_metrics(runs: list[dict]) -> list[tuple[str, str, float]]:
    return [
        spec for spec in METRICS
        if any(any(record.get(spec[0]) is not None and math.isfinite(record[spec[0]])
                   for record in run["records"])
               for run in runs)
    ]


def plot_individual(run: dict, output: Path, smooth_window: int):
    metrics = available_metrics([run])
    columns = 2
    rows = max(1, math.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.8 * rows),
                                squeeze=False, constrained_layout=True)
    axes = list(axes.flat)
    for axis, (key, title, scale) in zip(axes, metrics):
        xs, ys = metric_points(run["records"], key, scale)
        axis.plot(xs, ys, color="#9ecae1", linewidth=0.7, alpha=0.55, label="raw")
        axis.plot(xs, rolling(ys, smooth_window), color="#08519c", linewidth=1.5,
                  label=f"{smooth_window}-update mean")
        axis.set_title(title)
        axis.set_xlabel("Attempted training update")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for axis in axes[len(metrics):]:
        axis.axis("off")
    latest = run["records"][-1]
    figure.suptitle(
        f"{run['canonical_id']} — {run['status']} — "
        f"update {latest['iteration']}/{latest['total_iterations']}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=145)
    plt.close(figure)


def grouped_series(runs: list[dict], key: str, scale: float):
    by_iteration = {}
    for run in runs:
        for record in run["records"]:
            value = record.get(key)
            if value is not None and math.isfinite(value):
                by_iteration.setdefault(record["iteration"], []).append(value * scale)
    xs = sorted(by_iteration)
    means = [mean(by_iteration[item]) for item in xs]
    deviations = [stdev(by_iteration[item]) if len(by_iteration[item]) > 1 else 0.0
                  for item in xs]
    return xs, means, deviations


def plot_comparison(runs: list[dict], output: Path, smooth_window: int):
    metrics = available_metrics(runs)
    columns = 2
    rows = max(1, math.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.8 * rows),
                                squeeze=False, constrained_layout=True)
    axes = list(axes.flat)
    colors = plt.get_cmap("tab10")
    for axis, (key, title, scale) in zip(axes, metrics):
        for index, run in enumerate(sorted(runs, key=lambda item: item.get("seed", 0))):
            xs, ys = metric_points(run["records"], key, scale)
            if ys:
                axis.plot(xs, rolling(ys, smooth_window), linewidth=0.9, alpha=0.55,
                          color=colors(index), label=f"seed {run.get('seed', 'n/a')}")
        xs, means, deviations = grouped_series(runs, key, scale)
        if means:
            smooth_means = rolling(means, smooth_window)
            axis.plot(xs, smooth_means, color="black", linewidth=1.8, label="seed mean")
            lower = [value - dev for value, dev in zip(smooth_means, deviations)]
            upper = [value + dev for value, dev in zip(smooth_means, deviations)]
            axis.fill_between(xs, lower, upper, color="black", alpha=0.10,
                              label="±1 seed SD")
        axis.set_title(title)
        axis.set_xlabel("Attempted training update")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    for axis in axes[len(metrics):]:
        axis.axis("off")
    figure.suptitle(
        f"{runs[0]['task_profile']} / {runs[0]['result_group']} / "
        f"{display_comparison(runs[0]['comparison'])}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=145)
    plt.close(figure)


def plot_overview(runs: list[dict], output: Path, smooth_window: int):
    metrics = available_metrics(runs)
    figure, axes = plt.subplots(len(metrics), 1, figsize=(11, 3.3 * len(metrics)),
                               squeeze=False, constrained_layout=True)
    groups = {}
    for run in runs:
        groups.setdefault(run["comparison"], []).append(run)
    colors = plt.get_cmap("tab10")
    for axis, (key, title, scale) in zip(axes[:, 0], metrics):
        for index, (name, group) in enumerate(sorted(groups.items())):
            xs, means, deviations = grouped_series(group, key, scale)
            if not means:
                continue
            means = rolling(means, smooth_window)
            axis.plot(xs, means, linewidth=1.6, color=colors(index),
                      label=display_comparison(name))
            if len(group) > 1:
                axis.fill_between(
                    xs,
                    [value - dev for value, dev in zip(means, deviations)],
                    [value + dev for value, dev in zip(means, deviations)],
                    color=colors(index), alpha=0.10,
                )
        axis.set_title(title)
        axis.set_xlabel("Attempted training update")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    figure.suptitle(f"{runs[0]['task_profile']} / {runs[0]['result_group']} overview")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=145)
    plt.close(figure)


def load_runs(status: dict, campaign_dir: Path):
    jobs = list(status.get("reported_jobs", [])) + list(status.get("producers", []))
    histories = load_wandb_histories(campaign_dir)
    runs = []
    missing = []
    for job in jobs:
        log_value = job.get("log")
        log_path = Path(log_value) if log_value else None
        if log_path is None or not log_path.is_file():
            missing.append(job["canonical_id"])
            continue
        records = parse_metrics(log_path)
        if not records:
            missing.append(job["canonical_id"])
            continue
        merge_wandb(
            records,
            matching_wandb_history(job, histories),
            job.get("result_group", "calibration"),
        )
        item = dict(job)
        item["records"] = records
        item["comparison"] = comparison(item)
        if item.get("kind") == "producer":
            item["result_group"] = "calibration"
        runs.append(item)
    return runs, missing


def write_data_csv(path: Path, runs: list[dict]):
    fields = [
        "canonical_id", "task_profile", "result_group", "comparison", "seed",
        "status", "iteration", "total_iterations", "episodes", "checkpoint_score",
        "best_checkpoint_score", "success_rate", "best_success_rate", "cost",
        "completion_time_success_s", "terminal_time_residual_raw_s",
        "punctuality_mismatch_raw_s", "punctuality_mismatch_success_s",
        "remaining_time_success_s", "episode_metric_schema_version",
        "curriculum_ratio", "lagrange_multiplier", "max_episode_time",
        "best_max_episode_time",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            common = {key: run.get(key) for key in fields[:6]}
            for record in run["records"]:
                writer.writerow({**common, **{key: record.get(key) for key in fields[6:]}})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: CAMPAIGN_DIR/results)",
    )
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args(argv)
    if args.smooth_window < 1:
        parser.error("--smooth-window must be positive")

    campaign_dir = args.campaign_dir.resolve()
    status = json.loads((campaign_dir / "status.json").read_text())
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else campaign_dir / "results"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    runs, missing = load_runs(status, campaign_dir)
    charts = []

    for run in runs:
        path = (output_root / run["result_group"] / "training_curves" /
                run["task_profile"] / "individual" / f"{run['canonical_id']}.png")
        plot_individual(run, path, args.smooth_window)
        charts.append(str(path.relative_to(output_root)))

    by_comparison = {}
    by_task_group = {}
    for run in runs:
        group_key = (run["result_group"], run["task_profile"], run["comparison"])
        by_comparison.setdefault(group_key, []).append(run)
        task_key = (run["result_group"], run["task_profile"])
        by_task_group.setdefault(task_key, []).append(run)

    for (result_group, task, name), group in sorted(by_comparison.items()):
        path = (output_root / result_group / "training_curves" / task /
                f"grouped_{name}.png")
        plot_comparison(group, path, args.smooth_window)
        charts.append(str(path.relative_to(output_root)))

    for (result_group, task), group in sorted(by_task_group.items()):
        path = output_root / result_group / "training_curves" / task / "overview.png"
        plot_overview(group, path, args.smooth_window)
        charts.append(str(path.relative_to(output_root)))

    data_path = output_root / "training_curve_data.csv"
    write_data_csv(data_path, runs)
    manifest = {
        "campaign_id": status.get("campaign_id"),
        "campaign_status_snapshot_at": status.get("updated_at"),
        "generated_at": datetime.now().astimezone().isoformat(),
        "smooth_window_updates": args.smooth_window,
        "included_runs": len(runs),
        "missing_or_not_started_runs": missing,
        "missing_successful_completion_time_runs": [
            run["canonical_id"] for run in runs
            if not any(record.get("completion_time_success_s") is not None
                       for record in run["records"])
        ],
        "charts": sorted(charts),
        "data_csv": str(data_path.relative_to(output_root)),
        "note": "Running jobs are partial snapshots; rerun this script to refresh them.",
    }
    manifest_path = output_root / "training_curves_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    print(f"generated {len(charts)} charts for {len(runs)} available runs")


if __name__ == "__main__":
    main()
