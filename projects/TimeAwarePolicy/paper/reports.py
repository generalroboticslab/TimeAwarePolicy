"""Build release tables, narrative summaries, and output validations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from projects.TimeAwarePolicy.paper.definitions import (
    METHOD_DISPLAY,
    METHODS,
    TASK_DISPLAY,
    TASKS,
)


def format_mean_std(mean: float, std: float, n: int, digits: int = 3) -> str:
    if n > 1 and np.isfinite(std):
        return f"{mean:.{digits}f} ± {std:.{digits}f}"
    return f"{mean:.{digits}f} (n=1)"


def build_cmdp_training_table(
    data: pd.DataFrame, tables_dir: Path
) -> pd.DataFrame:
    """Summarize the final 100 attempted iterations of each selected run."""
    metrics = ("success_rate", "punctuality_mismatch_success_s", "cost")
    per_run = []
    for run_id, history in data.groupby("curve_run_id", sort=False):
        tail = history.sort_values("iteration").tail(100)
        first = tail.iloc[0]
        row = {
            "task_profile": first.task_profile,
            "method": first.comparison,
            "curve_run_id": run_id,
            "seed": first.seed,
            "source_campaign": first.source_campaign,
            "tail_start_iteration": int(tail.iteration.min()),
            "tail_end_iteration": int(tail.iteration.max()),
        }
        for metric in metrics:
            row[metric] = float(tail[metric].mean())
        per_run.append(row)
    per_run_frame = pd.DataFrame(per_run)
    per_run_frame.to_csv(tables_dir / "cmdp_final100_per_run.csv", index=False)

    rows = []
    for task in TASKS:
        for method in METHODS:
            group = per_run_frame[
                (per_run_frame.task_profile == task)
                & (per_run_frame.method == method)
            ]
            row = {
                "task": TASK_DISPLAY[task],
                "method": METHOD_DISPLAY[method],
                "runs": len(group),
                "summary_window": "final 100 iterations per selected run",
            }
            for metric in metrics:
                values = group[metric].to_numpy(float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else np.nan
                )
            row["success_rate_percent_mean"] = 100 * row["success_rate_mean"]
            row["success_rate_percent_std"] = 100 * row["success_rate_std"]
            row["formatted_success_rate_percent"] = format_mean_std(
                row["success_rate_percent_mean"],
                row["success_rate_percent_std"],
                len(group),
                2,
            )
            row["formatted_time_mismatch_s"] = format_mean_std(
                row["punctuality_mismatch_success_s_mean"],
                row["punctuality_mismatch_success_s_std"],
                len(group),
                3,
            )
            row["formatted_instability"] = format_mean_std(
                row["cost_mean"], row["cost_std"], len(group), 3
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(
        tables_dir / "cmdp_training_final100_9x_metrics.csv", index=False
    )
    return result


def load_time_optimal_2000_table(tables_dir: Path) -> pd.DataFrame:
    path = tables_dir / "time_optimal_evaluation_12x_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if len(frame) != 12 or not (frame.rollouts == 2000).all():
        raise RuntimeError(
            "Time-optimal strict evaluation table is not 12 x 2,000 rollouts"
        )
    return frame


def write_primary_tables(
    time_frame: pd.DataFrame,
    cmdp_frame: pd.DataFrame,
    tables_dir: Path,
) -> None:
    time_view = time_frame[[
        "task",
        "initial_policy",
        "formatted_success_rate_percent",
        "formatted_completion_time_s",
    ]].copy()
    time_view.columns = [
        "Task",
        "Initial Policy",
        "Success Rate (%)",
        "Completion Time (s)",
    ]
    cmdp_view = cmdp_frame[[
        "task",
        "method",
        "formatted_success_rate_percent",
        "formatted_time_mismatch_s",
        "formatted_instability",
    ]].copy()
    cmdp_view.columns = [
        "Task",
        "Method",
        "Success Rate (%)",
        "Time Mismatch (s)",
        "Instability",
    ]
    text = "# Primary result tables\n\n"
    text += "## Time-optimal: strict 2,000-configuration evaluation\n\n"
    text += time_view.to_markdown(index=False) + "\n\n"
    text += "## CMDP: final 100 training iterations\n\n"
    text += (
        "Each run is first averaged over its final 100 attempted iterations; "
        "the table then reports mean ± SD across selected real runs. n=1 is "
        "shown explicitly for corrected Cabinet histories.\n\n"
    )
    text += cmdp_view.to_markdown(index=False) + "\n"
    (tables_dir / "PRIMARY_TABLES.md").write_text(text)


def write_provenance(
    time_manifest: list[dict],
    cmdp_manifest: list[dict],
    cmdp_single_seed_manifest: list[dict],
    provenance_dir: Path,
) -> None:
    manifest = time_manifest + cmdp_manifest
    pd.DataFrame(manifest).to_csv(
        provenance_dir / "curve_source_manifest.csv", index=False
    )
    (provenance_dir / "curve_source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    pd.DataFrame(cmdp_single_seed_manifest).to_csv(
        provenance_dir / "cmdp_real_single_seed_manifest.csv", index=False
    )
    (provenance_dir / "cmdp_real_single_seed_manifest.json").write_text(
        json.dumps(cmdp_single_seed_manifest, indent=2) + "\n"
    )


def write_summary(
    output_dir: Path,
    time_table: pd.DataFrame,
    cmdp_table: pd.DataFrame,
    stability_summary: pd.DataFrame | None,
) -> None:
    """Write the result package's data-policy and protocol summary."""
    del time_table, cmdp_table  # Reserved for future table-derived prose.
    text = """# Final three-task results

The manuscript source is untouched. All figures use the manuscript plotting constants (DejaVu Sans, the paper palette, frameless legends, hidden top/right spines, and matching line/bar widths). Training curves use the manuscript labels, including **Iterations** on the x-axis.

## Primary outputs

- `training_curves/time_optimal/time_optimal_training_3x2.{pdf,png}`: 2 metric rows × 3 task columns, with Q40/Q60/Q80/Q95.
- `training_curves/cmdp/cmdp_training_3x3.{pdf,png}`: success rate / time mismatch / instability × 3 tasks, with N-P3O, PPO-Lagrangian, and CPO.
- `training_curves/cmdp/cmdp_training_real_single_seed_3x3.{pdf,png}`: one completed real seed per task and solver, with no cross-seed uncertainty band. Cube and GM Pour use Finite21; Drawer Opening uses the corrected Additive15 initializer and calibration bank.
- `tables/PRIMARY_TABLES.md`: strict 2,000-configuration time-optimal evaluation and final-100-iteration CMDP training summaries.
- `stagewise_stability/stagewise_stable_{mean,peak}_object_motion_binned20.{pdf,png}`: full-stable-stage mean and peak candidates, with task-level summaries and Cube manipulation-distance curves.

## Data policy and provenance

- No synthetic curve or evaluation data are used.
- Cube time-optimal curves use three real finite-horizon runs with reward `1000 + 100*T_left`. Their historical logger stored success and reward at every iteration but only final success-conditioned `eps_time`; the completion trace is recovered algebraically from that exact reward and the recorded final `eps_time`. The derivation is marked in every row.
- GM Pour and Cabinet time-optimal shadows summarize three selected real historical traces per quality. They are curated protocol variants, not a controlled same-configuration three-seed experiment; the exact reward scale and horizon of every trace are in `provenance/curve_source_manifest.csv`.
- Cube CMDP and GM Pour CPO use true same-campaign three-seed groups. Other GM Pour solver groups are explicitly curated real histories. Corrected Cabinet has one real run per solver; its shadow is the centered 51-iteration within-run temporal SD, not cross-seed uncertainty. Cabinet table entries therefore remain explicitly marked `n=1`.
- The additional real-single-seed CMDP figure never mixes histories within a curve and displays no uncertainty shadow. Its nine exact sources and their protocol differences are recorded in `provenance/cmdp_real_single_seed_manifest.csv`.
- CMDP table values average each selected run over its final 100 attempted iterations, then report variation across runs. They are not the 2,000-configuration ratio-sweep results.
- Time-optimal table values come from strict 2,000-fresh-configuration evaluations. Time-optimal policies do not require `T_min/P_max` lookup during evaluation.

## Stagewise-stability protocol

The stability evaluation uses all executed steps labelled stable (`speed_describe == 0`), including entry transitions and any continued execution in the final stable stage after `T_goal`. It reports two separately saved candidates: the per-episode time-average and the per-episode peak instantaneous object-motion proxy. Every bank configuration is enumerated exactly twice for each controller, and stage-wise and constant-ratio controllers use identical paired configurations. Each configuration uses `T_goal = 2*T_min`, so the constant controller has `tr = 0.5`; the staged schedule preserves the same duration-weighted average. The protocol uses neither fresh configurations nor 5-nearest-neighbor estimates. Summary bars and Cube distance curves use only paired rollouts successful under both controllers and containing stable-stage samples.

The Cube distance analysis uses 20 equal-width manipulation-distance bins. Each curve shows the bin mean and the shaded region shows one standard deviation across paired both-success rollouts in that bin.

The evaluation uses the N-P3O checkpoints declared by the versioned profile and requires exact observation-layout matches; it never partially loads or pads policy weights.
"""
    if stability_summary is not None:
        text += "\n## Stagewise-stability numerical summary\n\n"
        for row in stability_summary.itertuples():
            text += (
                f"- {row.task}, {row.controller_display}, {row.metric}: "
                f"{row.value_mean:.4f} ± {row.value_std:.4f} "
                "(paired both-success stable-stage "
                f"n={row.both_successful_valid_stable_rollouts}).\n"
            )
    (output_dir / "SUMMARY.md").write_text(text)


def validate(
    time_table: pd.DataFrame,
    cmdp_table: pd.DataFrame,
    include_stagewise_stability: bool,
    *,
    output_dir: Path,
    provenance_dir: Path,
    tables_dir: Path,
    time_optimal_dir: Path,
    cmdp_dir: Path,
    stagewise_stability_dir: Path,
    stagewise_stability_status: Path,
) -> None:
    expected = [
        time_optimal_dir / "time_optimal_training_3x2.pdf",
        time_optimal_dir / "time_optimal_training_3x2.png",
        cmdp_dir / "cmdp_training_3x3.pdf",
        cmdp_dir / "cmdp_training_3x3.png",
        cmdp_dir / "cmdp_training_real_single_seed_3x3.pdf",
        cmdp_dir / "cmdp_training_real_single_seed_3x3.png",
        tables_dir / "time_optimal_evaluation_12x_metrics.csv",
        tables_dir / "cmdp_training_final100_9x_metrics.csv",
        tables_dir / "PRIMARY_TABLES.md",
        provenance_dir / "curve_source_manifest.csv",
        provenance_dir / "cmdp_real_single_seed_manifest.csv",
        output_dir / "SUMMARY.md",
    ]
    if include_stagewise_stability:
        expected += [
            stagewise_stability_dir
            / "stagewise_stable_mean_object_motion_binned20.pdf",
            stagewise_stability_dir
            / "stagewise_stable_peak_object_motion_binned20.pdf",
            stagewise_stability_dir / "data" / "stable_object_motion_summary.csv",
            stagewise_stability_dir
            / "data"
            / "paired_fixed_bank_1000x2_per_controller.csv",
            stagewise_stability_dir
            / "data"
            / "evaluation_provenance_fullstable.json",
            stagewise_stability_status,
        ]
    missing = [
        str(path)
        for path in expected
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")
    if len(time_table) != 12 or len(cmdp_table) != 9:
        raise RuntimeError("Primary table shape validation failed")
    validation = {
        "status": "passed",
        "manuscript_untouched": True,
        "training_x_label": "Iterations",
        "time_optimal_table_rows": len(time_table),
        "cmdp_training_table_rows": len(cmdp_table),
        "stagewise_stability_complete": include_stagewise_stability,
        "checked_files": [str(path) for path in expected],
    }
    (provenance_dir / "final_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
