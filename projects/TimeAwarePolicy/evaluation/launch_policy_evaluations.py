#!/usr/bin/env python3
"""Launch the final real 2,000-configuration evaluations without retraining.

Time-optimal policies are evaluated once on 2,000 fresh configurations.
CMDP policies are evaluated at nine time ratios on fresh configurations; a
frozen 1,000-entry parent bank is used only for 5-NN estimates of T_min/P_max.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import cycle
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from projects.TimeAwarePolicy.paper.config import (
    load_profile,
    require_mapping,
    resolve_artifact,
    sha256_file,
)


DEFAULT_PROFILE = (
    ROOT
    / "projects"
    / "TimeAwarePolicy"
    / "paper"
    / "configs"
    / "policy_evaluation.json"
)
OUTPUT = ROOT / "results" / "final_three_task"
TRAIN_RES = ROOT / "train_res"
EVAL_RES = ROOT / "eval_res"
PYTHON = Path(sys.executable)
NUM_ENVS = 2000
WARMUP_EPISODES = 10000
INDEX = "best_rew"
SEED = 123456
GOAL_RATIO_RANGE = "[0.2, 1.0, 0.1]"
GPUS = tuple(range(8))
SLOTS_PER_GPU = 2

def slug(label: str) -> str:
    """Return a stable identifier for a public comparison label."""
    return label.lower().replace("-", "_").replace(" ", "_")


def selected_jobs(profile: dict, train_res: Path) -> list[dict]:
    """Resolve every checkpoint role declared by a policy-evaluation profile."""
    jobs = []
    tasks = require_mapping(profile, "tasks", "profile")
    for task_key, task_profile in tasks.items():
        task = task_profile.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError(f"profile.tasks.{task_key}.task must be a string")
        task_root = train_res / task
        time_optimal = require_mapping(
            task_profile, "time_optimal", f"profile.tasks.{task_key}"
        )
        for comparison, specification in time_optimal.items():
            training = resolve_artifact(
                task_root,
                specification,
                f"{task_key}.time_optimal.{comparison}",
            )
            jobs.append({
                "id": f"timeopt_{task_key}_{slug(comparison)}",
                "group": "time_optimal",
                "task_key": task_key,
                "task": task,
                "comparison": comparison,
                "training": str(training),
                "knn_fresh": False,
            })

        cmdp = require_mapping(
            task_profile, "cmdp", f"profile.tasks.{task_key}"
        )
        for method, specification in cmdp.items():
            training = resolve_artifact(
                task_root,
                specification,
                f"{task_key}.cmdp.{method}",
            )
            jobs.append({
                "id": f"cmdp_{task_key}_{slug(method)}",
                "group": "cmdp",
                "task_key": task_key,
                "task": task,
                "comparison": method,
                "training": str(training),
                "knn_fresh": True,
            })
    return jobs


def evaluation_name(training: Path, knn_fresh: bool) -> str:
    final_name = json.loads((training / "config.json").read_text())["final_name"]
    suffix = f"_EVAL_{INDEX}" + ("_KNNFresh5_CostV1" if knn_fresh else "")
    excess = max(0, len(final_name + suffix) - 250)
    return final_name[: len(final_name) - excess] + suffix


def command(job: dict) -> list[str]:
    training = Path(job["training"])
    cmd = [
        str(PYTHON),
        "-m",
        "projects.TimeAwarePolicy.eval",
        "--saving",
        "--task_name", job["task"],
        "--train_res_dir", str(TRAIN_RES),
        "--eval_res_dir", str(EVAL_RES),
        "--checkpoint", training.name,
        "--index_episode", INDEX,
        "--num_envs", str(NUM_ENVS),
        "--target_success_eps", str(NUM_ENVS),
        "--warmup_episodes", str(WARMUP_EPISODES),
        "--strict_eval",
        "--seed", str(SEED),
        "--graphics_device_id", "-1",
    ]
    if job["knn_fresh"]:
        cmd += [
            "--knn_configs_eval",
            "--constraint_cost_eval",
            "--goal_ratio_range", GOAL_RATIO_RANGE,
        ]
    return cmd


def validate_output(job: dict, output: Path) -> None:
    config_path = output / "config.json"
    metadata_path = output / "trajectories" / "meta_data.json"
    if not config_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"Incomplete evaluation output: {output}")
    cfg = json.loads(config_path.read_text())
    meta = json.loads(metadata_path.read_text())
    speed = meta.get("speed_and_time", {})
    expected_points = 9 if job["knn_fresh"] else 1
    required = {
        "time_ratio", "time_used", "time_goal", "time_mismatch",
        "max_inst", "sum_inst", "success_rate",
    }
    if job["knn_fresh"]:
        required.add("instability_cost")
    missing = sorted(required.difference(speed))
    wrong_lengths = {
        key: len(speed[key]) for key in required.difference(missing)
        if len(speed[key]) != expected_points
    }
    expected_cfg = {
        "checkpoint": Path(job["training"]).name,
        "index_episode": INDEX,
        "num_envs": NUM_ENVS,
        "strict_eval": True,
        "knn_configs_eval": job["knn_fresh"],
        "constraint_cost_eval": job["knn_fresh"],
    }
    mismatches = {}
    for key, value in expected_cfg.items():
        actual = cfg.get(key, False) if isinstance(value, bool) else cfg.get(key)
        if actual != value:
            mismatches[key] = {"actual": actual, "expected": value}
    if meta.get("episode") != NUM_ENVS or missing or wrong_lengths or mismatches:
        raise RuntimeError(
            f"Invalid evaluation {output}: episode={meta.get('episode')}, "
            f"missing={missing}, wrong_lengths={wrong_lengths}, mismatches={mismatches}"
        )


def sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    global OUTPUT, TRAIN_RES, EVAL_RES
    global NUM_ENVS, WARMUP_EPISODES, INDEX, SEED, GOAL_RATIO_RANGE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", type=Path, default=DEFAULT_PROFILE,
        help="Versioned policy-evaluation artifact profile.",
    )
    parser.add_argument("--train-res-dir", type=Path, default=TRAIN_RES)
    parser.add_argument("--eval-res-dir", type=Path, default=EVAL_RES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--gpus",
        default=",".join(map(str, GPUS)),
        help="Comma-separated CUDA device indices (default: 0,1,...,7).",
    )
    parser.add_argument(
        "--slots-per-gpu",
        type=int,
        default=SLOTS_PER_GPU,
        help="Maximum concurrent evaluator processes per GPU.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Resolve and validate every declared artifact without launching jobs.",
    )
    args = parser.parse_args()
    gpu_ids = tuple(int(item) for item in args.gpus.split(",") if item.strip())
    if not gpu_ids:
        parser.error("--gpus must contain at least one device index")
    if args.slots_per_gpu <= 0:
        parser.error("--slots-per-gpu must be positive")

    OUTPUT = args.output_dir.resolve()
    TRAIN_RES = args.train_res_dir.resolve()
    EVAL_RES = args.eval_res_dir.resolve()
    profile_path = args.profile.resolve()
    profile = load_profile(profile_path, "policy_evaluation")
    settings = require_mapping(profile, "evaluation", "profile")
    NUM_ENVS = int(settings["num_envs"])
    WARMUP_EPISODES = int(settings["warmup_episodes"])
    INDEX = str(settings["checkpoint_index"])
    SEED = int(settings["seed"])
    goal_ratio_range = settings["goal_ratio_range"]
    if (
        NUM_ENVS <= 0
        or WARMUP_EPISODES < 0
        or not isinstance(goal_ratio_range, list)
        or len(goal_ratio_range) != 3
    ):
        raise ValueError("Invalid evaluation settings in policy profile")
    GOAL_RATIO_RANGE = json.dumps(goal_ratio_range)

    jobs = selected_jobs(profile, TRAIN_RES)
    cmdp_gpu_order = cycle(gpu_ids)
    for index, job in enumerate(jobs):
        job["gpu"] = (
            next(cmdp_gpu_order) if job["group"] == "cmdp"
            else gpu_ids[index % len(gpu_ids)]
        )
        training = Path(job["training"])
        for required in (
            training / "config.json",
            training / "checkpoints" / f"eps_{INDEX}",
            training / "checkpoints" / f"rew_norm_eps_{INDEX}",
        ):
            if not required.is_file():
                raise FileNotFoundError(required)
        output = EVAL_RES / job["task"] / evaluation_name(
            training, job["knn_fresh"]
        )
        job["output"] = str(output)
        job["command"] = command(job)
        job["status"] = "pending"

    if args.preflight_only:
        print(json.dumps({
            "status": "passed",
            "profile": str(profile_path),
            "profile_sha256": sha256_file(profile_path),
            "jobs": [
                {key: job[key] for key in (
                    "id", "group", "task", "comparison", "training", "output"
                )}
                for job in jobs
            ],
        }, indent=2))
        return

    (OUTPUT / "logs" / "evaluation").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "provenance").mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "final_three_task_real_evaluation",
        "num_envs": NUM_ENVS,
        "warmup_episodes": WARMUP_EPISODES,
        "seed": SEED,
        "cmdp_time_ratios": [round(0.2 + 0.1 * i, 1) for i in range(9)],
        "cmdp_reference_rule": "fresh task configuration; mean T_min and P_max of 5 nearest frozen-bank entries",
        "cmdp_instability_metric": "success-conditioned undiscounted episode sum of max(p_t - p_max * tr, 0)",
        "max_processes_per_gpu": args.slots_per_gpu,
        "train_res_dir": str(TRAIN_RES),
        "eval_res_dir": str(EVAL_RES),
        "profile": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "source_sha256": sha256_files([
            ROOT / "core" / "evaluation" / "evaluator.py",
            ROOT / "projects" / "TimeAwarePolicy" / "eval.py",
            ROOT / "projects" / "TimeAwarePolicy" / "arguments" / "evaluation.py",
            ROOT / "envs" / "isaacgymenvs" / "tasks" / "base" / "vec_task.py",
            ROOT / "envs" / "isaacgymenvs" / "tasks" / "franka_cube_stack.py",
            ROOT / "envs" / "isaacgymenvs" / "tasks" / "franka_gm_pour.py",
            ROOT / "envs" / "isaacgymenvs" / "tasks" / "franka_cabinet.py",
        ]),
        "jobs": jobs,
    }
    manifest_path = OUTPUT / "provenance" / "evaluation_manifest.json"
    status_path = OUTPUT / "provenance" / "evaluation_status.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    lock = threading.Lock()
    semaphores = {
        gpu: threading.Semaphore(args.slots_per_gpu) for gpu in gpu_ids
    }

    def save_status() -> None:
        status_path.write_text(json.dumps({
            "updated_at": datetime.now().astimezone().isoformat(),
            "jobs": [{k: j.get(k) for k in ("id", "gpu", "status", "return_code", "output", "log")}
                     for j in jobs],
        }, indent=2, sort_keys=True) + "\n")

    def run(job: dict) -> str:
        output = Path(job["output"])
        log = OUTPUT / "logs" / "evaluation" / f"{job['id']}.log"
        job["log"] = str(log)
        with semaphores[job["gpu"]]:
            if output.exists():
                validate_output(job, output)
                with lock:
                    job["status"] = "reused"
                    job["return_code"] = 0
                    save_status()
                return job["id"]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
            env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (
                str(Path(sys.prefix) / "lib"),
                env.get("LD_LIBRARY_PATH"),
            )))
            env["PYTHONPATH"] = os.pathsep.join((
                str(ROOT / "isaacgym" / "python"),
                str(ROOT / "envs"),
                str(ROOT),
            ))
            with lock:
                job["status"] = "running"
                save_status()
            with log.open("w") as stream:
                result = subprocess.run(
                    job["command"], cwd=ROOT, env=env,
                    stdout=stream, stderr=subprocess.STDOUT,
                )
            with lock:
                job["return_code"] = result.returncode
                if result.returncode == 0:
                    try:
                        validate_output(job, output)
                    except Exception:
                        job["status"] = "failed"
                        save_status()
                        raise
                    else:
                        job["status"] = "completed"
                else:
                    job["status"] = "failed"
                save_status()
            if result.returncode != 0:
                raise RuntimeError(f"{job['id']} failed; see {log}")
            return job["id"]

    save_status()
    failures = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:  # keep independent jobs running
                failures.append({"job": futures[future]["id"], "error": str(exc)})
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
