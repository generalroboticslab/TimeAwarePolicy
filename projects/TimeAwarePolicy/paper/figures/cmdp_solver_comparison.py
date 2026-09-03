#!/usr/bin/env python3
"""Plot the matched real 27-run CMDP dataset and its GM Pour diagnostic.

The dataset consists of the 24-run campaign from 2026-08-25 plus the three
configuration-matched Cabinet seed-1 runs from the Additive15 campaign.  No
synthetic histories are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from projects.TimeAwarePolicy.paper.figures import training_curves as campaign_curves  # noqa: E402
from projects.TimeAwarePolicy.paper import curves as result_curves  # noqa: E402
from projects.TimeAwarePolicy.paper.style import (  # noqa: E402
    AxisLabelSize,
    BaselineLineWidth,
    TimeawareColor,
    TimeOptimalColor,
    VanillaColor,
    style_axis,
)
from projects.TimeAwarePolicy.paper.config import (  # noqa: E402
    load_profile,
    require_mapping,
    resolve_root_path,
    sha256_file,
)


DEFAULT_PROFILE = (
    ROOT
    / "projects"
    / "TimeAwarePolicy"
    / "paper"
    / "configs"
    / "figure_generation.json"
)
TASKS = ("cube", "gmpour", "cabinet")
METHODS = ("np3o", "ppo_lagrangian", "cpo")
SEEDS = (123456, 234567, 345678)
METRICS = ("success_rate", "punctuality_mismatch_success_s", "cost")


def load_cmdp24(campaign_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    status = json.loads((campaign_dir / "status.json").read_text())
    if status.get("result") != "complete" or status.get("failed_jobs"):
        raise RuntimeError("The 24-run campaign is not cleanly complete")
    runs, missing = campaign_curves.load_runs(status, campaign_dir)
    if missing or len(runs) != 24:
        raise RuntimeError(f"Expected 24 readable histories; got {len(runs)}, missing={missing}")

    pieces = []
    manifest = []
    for run in runs:
        frame = pd.DataFrame(run["records"])
        frame["canonical_id"] = run["canonical_id"]
        frame["task_profile"] = run["task_profile"]
        frame["comparison"] = run["method"]
        frame["seed"] = int(run["seed"])
        frame["status"] = run["status"]
        frame["curve_run_id"] = "cmdp24:" + run["canonical_id"]
        frame["source_campaign"] = campaign_dir.name
        frame["source_path"] = run["log"]
        frame["metric_origin"] = "direct training logger and offline W&B history"
        pieces.append(frame)
        manifest.append({
            "curve_run_id": frame["curve_run_id"].iloc[0],
            "task": run["task_profile"],
            "method": run["method"],
            "seed": int(run["seed"]),
            "source_campaign": campaign_dir.name,
            "source": run["log"],
        })
    return pd.concat(pieces, ignore_index=True, sort=False), manifest


def load_cabinet_seed1(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    frame = pd.read_csv(path)
    frame = frame[
        (frame.task_profile == "cabinet")
        & frame.comparison.isin(METHODS)
        & (frame.seed == SEEDS[0])
        & (frame.status == "completed")
    ].copy()
    frame["seed"] = frame.seed.astype(int)
    frame["curve_run_id"] = "additive15:" + frame.canonical_id.astype(str)
    frame["source_campaign"] = "cabinet_seed1"
    frame["source_path"] = str(path)
    frame["metric_origin"] = "direct training logger and offline W&B history"

    manifest = []
    for (run_id, method), history in frame.groupby(
        ["curve_run_id", "comparison"], sort=False,
    ):
        manifest.append({
            "curve_run_id": run_id,
            "task": "cabinet",
            "method": method,
            "seed": SEEDS[0],
            "source_campaign": "cabinet_seed1",
            "source": str(path),
        })
    return frame, manifest


def validate_dataset(data: pd.DataFrame) -> None:
    counts = data.groupby(["task_profile", "comparison"])["curve_run_id"].nunique()
    expected_index = pd.MultiIndex.from_product(
        [TASKS, METHODS], names=["task_profile", "comparison"],
    )
    counts = counts.reindex(expected_index, fill_value=0)
    if not (counts == 3).all():
        raise RuntimeError(f"Every task/method must have three runs:\n{counts}")

    iterations = data.groupby("curve_run_id").iteration.agg(["size", "min", "max"])
    bad = iterations[
        (iterations["size"] != 1500)
        | (iterations["min"] != 1)
        | (iterations["max"] != 1500)
    ]
    if not bad.empty:
        raise RuntimeError(f"Incomplete histories:\n{bad}")

    seeds = data.groupby(["task_profile", "comparison"]).seed.unique()
    for key, values in seeds.items():
        if set(map(int, values)) != set(SEEDS):
            raise RuntimeError(f"Seed mismatch for {key}: {values}")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_main_grid(data: pd.DataFrame, output_dir: Path) -> None:
    result_curves.plot_cmdp_grid(
        data,
        output_dir,
        stem="cmdp_training_real27_3x3",
        data_prefix="real27_",
        show_band=True,
    )


def plot_gmpour_np3o_seeds(data: pd.DataFrame, output_dir: Path) -> None:
    chosen = data[
        (data.task_profile == "gmpour") & (data.comparison == "np3o")
    ].copy()
    smoothed = result_curves.smooth_histories(chosen, METRICS)
    smoothed.to_csv(
        output_dir / "data" / "gmpour_np3o_three_seed_histories.csv", index=False,
    )

    colors = (TimeawareColor, VanillaColor, TimeOptimalColor)
    line_styles = ("-", "--", "-.")
    labels = {
        123456: "Seed 1 (123456)",
        234567: "Seed 2 (234567)",
        345678: "Seed 3 (345678)",
    }
    ylabels = ("Success Rate", "Time Mismatch (s)", "Instability")
    fig, axes = plt.subplots(3, 1, figsize=(11, 18), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for seed, color, line_style in zip(SEEDS, colors, line_styles):
        history = smoothed[smoothed.seed == seed].sort_values("iteration")
        for axis, metric in zip(axes, METRICS):
            axis.plot(
                history.iteration,
                history[metric],
                color=color,
                linestyle=line_style,
                linewidth=BaselineLineWidth,
                label=labels[seed],
            )

    axes[0].set_title(
        "Granular Media Pouring — N-P3O", pad=12, fontsize=AxisLabelSize,
    )
    axes[0].set_ylim(-0.02, 1.02)
    for index, (axis, ylabel) in enumerate(zip(axes, ylabels)):
        axis.set_xlim(0, 1500)
        axis.set_ylabel(ylabel)
        if index == len(axes) - 1:
            axis.set_xlabel("Iterations")
        style_axis(axis)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=AxisLabelSize,
        columnspacing=1.0,
        handlelength=1.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_dir / "gmpour_np3o_three_seeds_3x1")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", type=Path, default=DEFAULT_PROFILE,
        help="Versioned final-result input profile.",
    )
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument("--cabinet-seed1-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    profile_path = args.profile.resolve()
    profile = load_profile(profile_path, "figure_generation")
    comparison = require_mapping(profile, "cmdp_solver_comparison", "profile")
    campaign_dir = (
        args.campaign_dir.resolve()
        if args.campaign_dir is not None
        else resolve_root_path(ROOT, str(comparison["campaign_dir"]))
    )
    cabinet_seed1_csv = (
        args.cabinet_seed1_csv.resolve()
        if args.cabinet_seed1_csv is not None
        else resolve_root_path(ROOT, str(comparison["cabinet_seed1_csv"]))
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else resolve_root_path(ROOT, str(comparison["output_dir"]))
    )
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    cmdp24, current_manifest = load_cmdp24(campaign_dir)
    cabinet, cabinet_manifest = load_cabinet_seed1(cabinet_seed1_csv)
    selected = pd.concat([cmdp24, cabinet], ignore_index=True, sort=False)
    validate_dataset(selected)

    selected.to_csv(
        output_dir / "data" / "cmdp_solver_comparison_histories.csv",
        index=False,
    )
    manifest = {
        "dataset": "matched real CMDP 27-run dataset",
        "synthetic_histories": False,
        "run_count": int(selected.curve_run_id.nunique()),
        "task_method_groups": 9,
        "runs_per_group": 3,
        "iterations_per_run": 1500,
        "smoothing": "centered 25-iteration rolling mean",
        "uncertainty": "sample standard deviation across three seeds",
        "profile": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "sources": current_manifest + cabinet_manifest,
    }
    (output_dir / "data" / "cmdp_solver_comparison_provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    plot_main_grid(selected, output_dir)
    plot_gmpour_np3o_seeds(selected, output_dir)
    print(json.dumps({
        "runs": manifest["run_count"],
        "main_plot": str(output_dir / "cmdp_training_real27_3x3.pdf"),
        "diagnostic_plot": str(output_dir / "gmpour_np3o_three_seeds_3x1.pdf"),
    }, indent=2))


if __name__ == "__main__":
    main()
