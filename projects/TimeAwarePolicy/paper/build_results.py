#!/usr/bin/env python3
"""Complete the final three-task paper result package from real stored data.

This script never edits the manuscript.  It builds manuscript-styled training
    curves, scalar tables, and paired stagewise-stability panels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from projects.TimeAwarePolicy.paper.config import (
    load_profile,
    require_mapping,
    resolve_root_path,
    sha256_file,
)
from projects.TimeAwarePolicy.paper.figures import (
    stagewise_stability as stagewise_stability_results,
)
from projects.TimeAwarePolicy.paper import curves as result_curves
from projects.TimeAwarePolicy.paper import datasets as result_datasets
from projects.TimeAwarePolicy.paper import reports as result_reports


DEFAULT_PROFILE = (
    ROOT
    / "projects"
    / "TimeAwarePolicy"
    / "paper"
    / "configs"
    / "figure_generation.json"
)
ACTIVE_PROFILE_PATH = DEFAULT_PROFILE
ACTIVE_PROFILE = load_profile(DEFAULT_PROFILE, "figure_generation")
STABILITY_EVALUATION_PROFILE_PATH = resolve_root_path(
    ROOT, ACTIVE_PROFILE["stagewise_stability_evaluation_profile"]
)
STABILITY_EVALUATION_PROFILE = load_profile(
    STABILITY_EVALUATION_PROFILE_PATH, "stagewise_stability_evaluation"
)
OUT = resolve_root_path(ROOT, ACTIVE_PROFILE["default_output_dir"])
PROV = OUT / "provenance"
TABLES = OUT / "tables"
TIMEOPT = OUT / "training_curves" / "time_optimal"
CMDP = OUT / "training_curves" / "cmdp"
STABILITY = OUT / "stagewise_stability"
STABILITY_STATUS = (
    STABILITY / "evaluation_campaign" / "full_fullstable_repeat2_status.json"
)

CURVE_INPUTS = {
    name: OUT / relative_path
    for name, relative_path in require_mapping(
        ACTIVE_PROFILE, "curve_inputs", "profile"
    ).items()
}
REVIEW_CUBE = resolve_root_path(ROOT, ACTIVE_PROFILE["review_cube_dir"])


def configure_paths(
    output_dir: Path,
    profile: dict,
    profile_path: Path,
    review_cube: Path | None = None,
) -> None:
    """Configure all derived output/input paths from public CLI arguments."""
    global OUT, PROV, TABLES, TIMEOPT, CMDP, STABILITY, STABILITY_STATUS
    global CURVE_INPUTS, REVIEW_CUBE, ACTIVE_PROFILE, ACTIVE_PROFILE_PATH
    global STABILITY_EVALUATION_PROFILE_PATH, STABILITY_EVALUATION_PROFILE
    ACTIVE_PROFILE = profile
    ACTIVE_PROFILE_PATH = profile_path.resolve()
    STABILITY_EVALUATION_PROFILE_PATH = resolve_root_path(
        ROOT, str(profile["stagewise_stability_evaluation_profile"])
    )
    STABILITY_EVALUATION_PROFILE = load_profile(
        STABILITY_EVALUATION_PROFILE_PATH, "stagewise_stability_evaluation"
    )
    OUT = output_dir.resolve()
    PROV = OUT / "provenance"
    TABLES = OUT / "tables"
    TIMEOPT = OUT / "training_curves" / "time_optimal"
    CMDP = OUT / "training_curves" / "cmdp"
    STABILITY = OUT / "stagewise_stability"
    STABILITY_STATUS = OUT / str(profile["stagewise_stability_status"])
    CURVE_INPUTS = {
        name: OUT / str(relative_path)
        for name, relative_path in require_mapping(
            profile, "curve_inputs", "profile"
        ).items()
    }
    REVIEW_CUBE = (
        review_cube.resolve()
        if review_cube is not None
        else resolve_root_path(ROOT, str(profile["review_cube_dir"]))
    )

def prepare_directories() -> None:
    for directory in (PROV, TABLES, TIMEOPT, CMDP, STABILITY):
        directory.mkdir(parents=True, exist_ok=True)
    for directory in (TIMEOPT / "data", CMDP / "data", STABILITY / "data"):
        directory.mkdir(parents=True, exist_ok=True)
    scripts = OUT / "scripts"
    scripts.mkdir(exist_ok=True)
    for source in (
        Path(__file__),
        ROOT
        / "projects"
        / "TimeAwarePolicy"
        / "evaluation"
        / "launch_stagewise_stability.py",
        ROOT
        / "projects"
        / "TimeAwarePolicy"
        / "evaluation"
        / "validate_stagewise_stability.py",
    ):
        shutil.copy2(source, scripts / source.name)
    write_profile_provenance(PROV)


def write_profile_provenance(directory: Path) -> None:
    """Record the versioned input profile next to generated outputs."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result_profile.json").write_text(json.dumps({
        "profile_path": str(ACTIVE_PROFILE_PATH),
        "profile_sha256": sha256_file(ACTIVE_PROFILE_PATH),
        "profile": ACTIVE_PROFILE,
    }, indent=2, sort_keys=True) + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", type=Path, default=DEFAULT_PROFILE,
        help="Versioned input profile for the available final-result package.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Result package root (default comes from --profile).",
    )
    parser.add_argument(
        "--review-cube-dir",
        type=Path,
        default=None,
        help="Historical Cube campaign directory used by the time-optimal curves.",
    )
    parser.add_argument(
        "--include-stagewise-stability",
        dest="include_stagewise_stability",
        action="store_true",
        help="Require completed paired stagewise-stability evaluations.",
    )
    parser.add_argument(
        "--stagewise-stability-only",
        dest="stagewise_stability_only",
        action="store_true",
        help="Build and validate only the stagewise-stability analysis.",
    )
    parser.add_argument(
        "--stagewise-stability-status",
        dest="stagewise_stability_status",
        type=Path,
        help="Completed stagewise-stability evaluation status JSON.",
    )
    parser.add_argument(
        "--stagewise-distance-bins",
        dest="stagewise_distance_bins",
        type=int,
        default=20,
        help="Plot binned mean curves with SD bands (default: 20 bins).",
    )
    args = parser.parse_args(argv)
    profile_path = args.profile.resolve()
    profile = load_profile(profile_path, "figure_generation")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else resolve_root_path(ROOT, str(profile["default_output_dir"]))
    )
    configure_paths(output_dir, profile, profile_path, args.review_cube_dir)
    if args.stagewise_stability_status is None:
        args.stagewise_stability_status = STABILITY_STATUS

    if args.stagewise_stability_only:
        STABILITY.mkdir(parents=True, exist_ok=True)
        (STABILITY / "data").mkdir(parents=True, exist_ok=True)
        write_profile_provenance(STABILITY / "data")
        scripts = OUT / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        for source in (
            Path(__file__),
            ROOT
            / "projects"
            / "TimeAwarePolicy"
            / "evaluation"
            / "launch_stagewise_stability.py",
            ROOT
            / "projects"
            / "TimeAwarePolicy"
            / "evaluation"
            / "validate_stagewise_stability.py",
        ):
            shutil.copy2(source, scripts / source.name)
        stability_summary = stagewise_stability_results.build(
            status_path=args.stagewise_stability_status,
            output_dir=STABILITY,
            evaluation_profile=STABILITY_EVALUATION_PROFILE,
            distance_bins=args.stagewise_distance_bins,
        )
        result_reports.write_summary(
            OUT, pd.DataFrame(), pd.DataFrame(), stability_summary
        )
        stagewise_stability_results.validate_outputs(
            summary=stability_summary,
            status_path=args.stagewise_stability_status,
            output_dir=STABILITY,
            distance_bins=args.stagewise_distance_bins,
            evaluation_profile=STABILITY_EVALUATION_PROFILE,
        )
        return

    prepare_directories()
    frames = result_datasets.load_curve_inputs(CURVE_INPUTS)
    time_data, time_manifest = result_datasets.select_time_optimal(
        frames, REVIEW_CUBE
    )
    cmdp_data, cmdp_manifest = result_datasets.select_cmdp(frames)
    cmdp_single_seed_data, cmdp_single_seed_manifest = (
        result_datasets.select_cmdp_real_single_seed(frames)
    )
    result_reports.write_provenance(
        time_manifest,
        cmdp_manifest,
        cmdp_single_seed_manifest,
        PROV,
    )
    result_curves.plot_time_optimal(time_data, TIMEOPT)
    result_curves.plot_cmdp(cmdp_data, CMDP)
    result_curves.plot_cmdp_real_single_seed(cmdp_single_seed_data, CMDP)
    time_table = result_reports.load_time_optimal_2000_table(TABLES)
    cmdp_table = result_reports.build_cmdp_training_table(cmdp_data, TABLES)
    result_reports.write_primary_tables(time_table, cmdp_table, TABLES)
    stability_summary = (
        stagewise_stability_results.build(
            status_path=args.stagewise_stability_status,
            output_dir=STABILITY,
            evaluation_profile=STABILITY_EVALUATION_PROFILE,
            distance_bins=args.stagewise_distance_bins,
        )
        if args.include_stagewise_stability else None
    )
    result_reports.write_summary(
        OUT, time_table, cmdp_table, stability_summary
    )
    result_reports.validate(
        time_table,
        cmdp_table,
        args.include_stagewise_stability,
        output_dir=OUT,
        provenance_dir=PROV,
        tables_dir=TABLES,
        time_optimal_dir=TIMEOPT,
        cmdp_dir=CMDP,
        stagewise_stability_dir=STABILITY,
        stagewise_stability_status=args.stagewise_stability_status,
    )
    print(OUT)


if __name__ == "__main__":
    main()
