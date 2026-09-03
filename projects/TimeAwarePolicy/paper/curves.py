"""Training-curve aggregation and plotting for released result packages."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from projects.TimeAwarePolicy.paper.definitions import (
    METHOD_COLORS,
    METHOD_DISPLAY,
    METHODS,
    QUALITIES,
    QUALITY_COLORS,
    QUALITY_DISPLAY,
    TASK_DISPLAY,
    TASKS,
)
from projects.TimeAwarePolicy.paper.style import (
    AxisLabelSize,
    BaselineLineWidth,
    LegendSize,
    style_axis,
)


def smooth_histories(data: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    """Apply released smoothing and retain honest within-run variability."""
    pieces = []
    for _, history in data.groupby("curve_run_id", sort=False):
        history = history.sort_values("iteration").copy()
        for metric in metrics:
            raw = history[metric].copy()
            history[f"{metric}_within_run_std"] = raw.rolling(
                51, min_periods=10, center=True,
            ).std(ddof=1)
            history[metric] = raw.rolling(
                25, min_periods=1, center=True,
            ).mean()
        pieces.append(history)
    return pd.concat(pieces, ignore_index=True, sort=False)


def aggregate_curves(
    data: pd.DataFrame,
    comparison: str,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    """Aggregate real histories and label the uncertainty source precisely."""
    chosen = data[data.comparison == comparison]
    run_count = chosen["curve_run_id"].nunique()
    rows = []
    for iteration, group in chosen.groupby("iteration", sort=True):
        row = {"iteration": int(iteration), "comparison": comparison}
        for metric in metrics:
            values = group[metric].dropna().to_numpy(float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            if run_count > 1:
                row[f"{metric}_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else np.nan
                )
                row[f"{metric}_uncertainty_type"] = "across_selected_run_sd"
            else:
                local = group[f"{metric}_within_run_std"].dropna().to_numpy(float)
                row[f"{metric}_std"] = float(local.mean()) if len(local) else np.nan
                row[f"{metric}_uncertainty_type"] = (
                    "centered_51_iteration_within_run_temporal_sd"
                )
            row[f"{metric}_n"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def draw_curve(
    axis: plt.Axes,
    aggregate: pd.DataFrame,
    metric: str,
    label: str,
    color: str,
    show_band: bool = True,
) -> None:
    x = aggregate.iteration.to_numpy(float)
    mean = aggregate[f"{metric}_mean"].to_numpy(float)
    std = aggregate[f"{metric}_std"].fillna(0).to_numpy(float)
    axis.plot(x, mean, color=color, linewidth=BaselineLineWidth, label=label)
    valid = np.isfinite(mean) & np.isfinite(std)
    if show_band and valid.any():
        lower = np.maximum(mean[valid] - std[valid], 0.0)
        upper = mean[valid] + std[valid]
        if metric == "success_rate":
            upper = np.minimum(upper, 1.0)
        axis.fill_between(
            x[valid], lower, upper, color=color, alpha=0.18, linewidth=0,
        )


def save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        stem.with_suffix(".pdf"), bbox_inches="tight", dpi=300, transparent=False
    )
    figure.savefig(
        stem.with_suffix(".png"), bbox_inches="tight", dpi=300, transparent=False
    )
    plt.close(figure)


def plot_time_optimal(data: pd.DataFrame, output_dir: Path) -> None:
    metrics = ("success_rate", "completion_time_success_s")
    smoothed = smooth_histories(data, metrics)
    smoothed.to_csv(output_dir / "data" / "selected_real_histories.csv", index=False)
    aggregates = []
    ylabels = ("Success Rate", "Completion Time (s)")
    figure, axes = plt.subplots(2, 3, figsize=(25, 13), sharex=True, squeeze=False)
    for column, task in enumerate(TASKS):
        task_data = smoothed[smoothed.task_profile == task]
        for quality in QUALITIES:
            aggregate = aggregate_curves(task_data, quality, metrics)
            aggregate["task"] = task
            aggregates.append(aggregate)
            for row, metric in enumerate(metrics):
                draw_curve(
                    axes[row, column], aggregate, metric,
                    QUALITY_DISPLAY[quality], QUALITY_COLORS[quality],
                )
        axes[0, column].set_title(
            TASK_DISPLAY[task], pad=12, fontsize=AxisLabelSize,
        )
        axes[0, column].set_ylim(-0.02, 1.02)
        for row, ylabel in enumerate(ylabels):
            axes[row, column].set_ylabel(ylabel if column == 0 else "")
            axes[row, column].set_xlabel("Iterations" if row == 1 else "")
            axes[row, column].set_xlim(0, 1500)
            style_axis(axes[row, column])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=4,
        bbox_to_anchor=(0.5, 1.015), fontsize=AxisLabelSize,
        columnspacing=1.1, handlelength=1.4,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, output_dir / "time_optimal_training_3x2")
    pd.concat(aggregates, ignore_index=True).to_csv(
        output_dir / "data" / "mean_std_curves.csv", index=False,
    )


def plot_cmdp_grid(
    data: pd.DataFrame,
    output_dir: Path,
    stem: str,
    data_prefix: str,
    show_band: bool,
) -> None:
    metrics = ("success_rate", "punctuality_mismatch_success_s", "cost")
    ylabels = ("Success Rate", "Time Mismatch (s)", "Instability")
    smoothed = smooth_histories(data, metrics)
    smoothed.to_csv(
        output_dir / "data" / f"{data_prefix}selected_real_histories.csv",
        index=False,
    )
    aggregates = []
    figure, axes = plt.subplots(3, 3, figsize=(25, 19), sharex=True, squeeze=False)
    for column, task in enumerate(TASKS):
        task_data = smoothed[smoothed.task_profile == task]
        for method in METHODS:
            aggregate = aggregate_curves(task_data, method, metrics)
            aggregate["task"] = task
            aggregates.append(aggregate)
            for row, metric in enumerate(metrics):
                draw_curve(
                    axes[row, column], aggregate, metric,
                    METHOD_DISPLAY[method], METHOD_COLORS[method], show_band,
                )
        axes[0, column].set_title(
            TASK_DISPLAY[task], pad=12, fontsize=AxisLabelSize,
        )
        axes[0, column].set_ylim(-0.02, 1.02)
        for row, ylabel in enumerate(ylabels):
            axes[row, column].set_ylabel(ylabel if column == 0 else "")
            axes[row, column].set_xlabel("Iterations" if row == 2 else "")
            axes[row, column].set_xlim(0, 1500)
            style_axis(axes[row, column])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=3,
        bbox_to_anchor=(0.5, 1.015), fontsize=AxisLabelSize,
        columnspacing=1.1, handlelength=1.4,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, output_dir / stem)
    pd.concat(aggregates, ignore_index=True).to_csv(
        output_dir / "data" / f"{data_prefix}mean_std_curves.csv", index=False,
    )


def plot_cmdp(data: pd.DataFrame, output_dir: Path) -> None:
    plot_cmdp_grid(
        data,
        output_dir,
        stem="cmdp_training_3x3",
        data_prefix="",
        show_band=True,
    )


def plot_cmdp_real_single_seed(data: pd.DataFrame, output_dir: Path) -> None:
    plot_cmdp_grid(
        data,
        output_dir,
        stem="cmdp_training_real_single_seed_3x3",
        data_prefix="real_single_seed_",
        show_band=False,
    )
