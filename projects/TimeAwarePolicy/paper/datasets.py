"""Load and select the real training histories used by release figures."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from projects.TimeAwarePolicy.paper.definitions import METHODS, QUALITIES, TASKS


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TRAIN_RE = re.compile(
    rf"Current Iteration:\s*(\d+)/(\d+)\s*\|\s*Episodes:\s*(\d+)\s*\|\s*"
    rf"Reward:\s*({NUMBER})/(?:{NUMBER}|-inf)\s*\|\s*"
    rf"Success Rate:\s*({NUMBER})/(?:{NUMBER})"
)


def load_curve_inputs(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Load named campaign tables and attach stable source identifiers."""
    frames = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["source_campaign"] = name
        frame["source_path"] = str(path)
        frame["curve_run_id"] = name + ":" + frame["canonical_id"].astype(str)
        frame["metric_origin"] = "direct training logger"
        frames[name] = frame
    return frames


def pick(
    frame: pd.DataFrame, task: str, comparison: str, seed: int
) -> pd.DataFrame:
    """Select one required run or fail with an actionable error."""
    chosen = frame[
        (frame.task_profile == task)
        & (frame.comparison == comparison)
        & (frame.seed == seed)
    ].copy()
    if chosen.empty:
        raise ValueError(f"Missing history {task}/{comparison}/seed={seed}")
    return chosen


def load_review_cube_histories(
    review_cube: Path,
) -> tuple[pd.DataFrame, list[dict]]:
    """Recover the historical Cube time-optimal traces and their provenance."""
    summary = pd.read_csv(review_cube / "analysis" / "run_summary.csv")
    pieces = []
    manifest = []
    for row in summary.itertuples():
        match = re.fullmatch(r"initq(40|60|80|95)_timeopt_s([123])", row.name)
        if not match:
            continue
        quality = f"q{match.group(1)}"
        seed_index = int(match.group(2))
        seed = (123456, 234567, 345678)[seed_index - 1]
        log_path = review_cube / "logs" / f"{row.name}.log"
        text = log_path.read_text(errors="replace").replace("\r", "\n")
        records = []
        for found in TRAIN_RE.finditer(text):
            iteration, total, episodes, reward, success = found.groups()
            records.append({
                "iteration": int(iteration),
                "total_iterations": int(total),
                "episodes": int(episodes),
                "checkpoint_score": float(reward),
                "success_rate": float(success),
            })
        if len(records) != 1500:
            raise RuntimeError(
                f"{log_path} contains {len(records)} iterations, expected 1500"
            )
        frame = pd.DataFrame(records)
        last = frame.iloc[-1]
        if last.success_rate <= 0:
            raise RuntimeError(
                f"Cannot calibrate successful completion time for {row.name}"
            )
        # The historical logger retained per-iteration mean terminal reward but
        # only the final success-conditioned completion time. Under the exact
        # reward 1000 + 100*T_left, reward/success gives mean successful
        # terminal reward. Calibrate the temporal horizon at the final point,
        # then recover the completion-time trace algebraically.
        mean_horizon_s = float(row.metadata_episode_time) + (
            float(last.checkpoint_score) / float(last.success_rate) - 1000.0
        ) / 100.0
        valid = frame.success_rate > 0
        completion = pd.Series(np.nan, index=frame.index, dtype=float)
        completion.loc[valid] = mean_horizon_s - (
            frame.loc[valid, "checkpoint_score"]
            / frame.loc[valid, "success_rate"]
            - 1000.0
        ) / 100.0
        completion.loc[(completion < 0) | (completion > 20)] = np.nan
        frame["completion_time_success_s"] = completion
        frame["task_profile"] = "cube"
        frame["comparison"] = quality
        frame["seed"] = seed
        frame["canonical_id"] = row.name
        frame["curve_run_id"] = f"review_cube:{row.name}"
        frame["source_campaign"] = "review_cube"
        frame["source_path"] = str(log_path)
        frame["metric_origin"] = (
            "success direct; completion derived from logged reward/success "
            "under 1000+100*T_left and calibrated final successful eps_time"
        )
        pieces.append(frame)
        manifest.append({
            "plot_group": "time_optimal",
            "task": "cube",
            "comparison": quality,
            "seed": seed,
            "canonical_id": row.name,
            "source_campaign": "review_cube",
            "reward": "1000 + 100*T_left",
            "horizon": "finite",
            "replication_class": "same_campaign_seed_replication",
            "completion_metric": "algebraically recovered real logged metric",
            "source": str(log_path),
        })
    return pd.concat(pieces, ignore_index=True), manifest


def select_time_optimal(
    frames: dict[str, pd.DataFrame], review_cube: Path
) -> tuple[pd.DataFrame, list[dict]]:
    """Select the real three-trace histories used for time-optimal plots."""
    cube, manifest = load_review_cube_histories(review_cube)
    pieces = [cube]

    # The corrected additive campaign supplies one exact 100-scale run for GM
    # Pour and Cabinet. Two real successful historical runs are added to make
    # the requested three-trace visualization. This is curated historical
    # variation, not a controlled matched-seed study.
    for task in ("gmpour", "cabinet"):
        for quality in QUALITIES:
            selections = [("additive15", 123456)]
            if task == "gmpour" and quality == "q60":
                selections += [("corrected", 234567), ("corrected", 345678)]
            else:
                selections += [("full3", 234567), ("full3", 345678)]
            for campaign, seed in selections:
                part = pick(frames[campaign], task, quality, seed)
                part = part[part.iteration < 1500].copy()
                pieces.append(part)
                scale = {
                    "additive15": 100,
                    "full3": 200,
                    "corrected": 1000,
                }[campaign]
                manifest.append({
                    "plot_group": "time_optimal",
                    "task": task,
                    "comparison": quality,
                    "seed": seed,
                    "canonical_id": part.canonical_id.iloc[0],
                    "source_campaign": campaign,
                    "reward": f"1000 + {scale}*T_left",
                    "horizon": (
                        "finite" if campaign == "additive15" else "infinite"
                    ),
                    "replication_class": "curated_real_historical_run",
                    "completion_metric": (
                        "direct success-conditioned training logger"
                    ),
                    "source": part.source_path.iloc[0],
                })
    selected = pd.concat(pieces, ignore_index=True, sort=False)
    counts = selected.groupby(
        ["task_profile", "comparison"]
    )["curve_run_id"].nunique()
    if not (counts == 3).all():
        raise RuntimeError(
            f"Time-optimal groups are not all three real traces:\n{counts}"
        )
    return selected, manifest


def select_cmdp(
    frames: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, list[dict]]:
    """Select real histories for the multi-run CMDP comparison."""
    pieces = []
    manifest = []

    def add(
        campaign: str,
        task: str,
        method: str,
        seed: int,
        classification: str,
    ) -> None:
        part = pick(frames[campaign], task, method, seed)
        part = part[part.iteration < 1500].copy()
        pieces.append(part)
        manifest.append({
            "plot_group": "cmdp",
            "task": task,
            "comparison": method,
            "seed": seed,
            "canonical_id": part.canonical_id.iloc[0],
            "source_campaign": campaign,
            "replication_class": classification,
            "curve_uncertainty": (
                "centered 51-iteration within-run temporal SD"
                if classification == "single_corrected_real_run"
                else "across-selected-run SD"
            ),
            "source": part.source_path.iloc[0],
        })

    for method in METHODS:
        for seed in (123456, 234567, 345678):
            add(
                "full3", "cube", method, seed,
                "same_campaign_seed_replication",
            )

    gm = {
        "np3o": (("full3", 123456), ("corrected", 234567), ("finite21", 123456)),
        "ppo_lagrangian": (
            ("full3", 123456),
            ("full3", 345678),
            ("corrected", 234567),
        ),
        "cpo": (("full3", 123456), ("full3", 234567), ("full3", 345678)),
    }
    for method, histories in gm.items():
        for campaign, seed in histories:
            classification = (
                "same_campaign_seed_replication"
                if method == "cpo"
                else "curated_real_historical_run"
            )
            add(campaign, "gmpour", method, seed, classification)

    for method in METHODS:
        add(
            "additive15", "cabinet", method, 123456,
            "single_corrected_real_run",
        )

    return pd.concat(pieces, ignore_index=True, sort=False), manifest


def select_cmdp_real_single_seed(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[dict]]:
    """Select one completed real run per task and solver, without seed mixing."""
    pieces = []
    manifest = []
    # Finite21 is the latest complete one-seed, three-solver campaign for Cube
    # and GM Pour. Cabinet from that campaign used an incompatible initializer,
    # so use the later corrected Additive15 Cabinet run instead.
    campaign_by_task = {
        "cube": "finite21",
        "gmpour": "finite21",
        "cabinet": "additive15",
    }
    protocol_by_campaign = {
        "finite21": {
            "horizon": "finite",
            "critic_reset": False,
            "warmup_iterations": 0,
            "punctuality_scale": 100,
            "target_kl": 5.0,
            "scene_velocity_reward_scale": "[50, 50] (inactive with use_cost=true)",
            "exponential_velocity_schedule": True,
        },
        "additive15": {
            "horizon": "finite",
            "critic_reset": True,
            "warmup_iterations": 50,
            "punctuality_scale": 100,
            "target_kl": 2.5,
            "scene_velocity_reward_scale": "[0, 0]",
            "exponential_velocity_schedule": False,
        },
    }
    for task in TASKS:
        campaign = campaign_by_task[task]
        for method in METHODS:
            part = pick(frames[campaign], task, method, 123456)
            part = part[part.iteration <= 1500].copy()
            if part.status.iloc[0] != "completed" or len(part) != 1500:
                raise RuntimeError(
                    "Single-seed source is not a complete 1,500-iteration "
                    f"run: {task}/{method}/{campaign}"
                )
            pieces.append(part)
            manifest.append({
                "task": task,
                "comparison": method,
                "seed": 123456,
                "canonical_id": part.canonical_id.iloc[0],
                "source_campaign": campaign,
                "status": "completed",
                "iterations": 1500,
                "uncertainty": "none; one real run",
                "fixed_configuration_bank": True,
                "time_ratio": True,
                **protocol_by_campaign[campaign],
                "source": part.source_path.iloc[0],
            })
    selected = pd.concat(pieces, ignore_index=True, sort=False)
    counts = selected.groupby(
        ["task_profile", "comparison"]
    )["curve_run_id"].nunique()
    if len(counts) != 9 or not (counts == 1).all():
        raise RuntimeError(
            "Single-seed CMDP selection is not 3 tasks x 3 methods:\n"
            f"{counts}"
        )
    return selected, manifest
