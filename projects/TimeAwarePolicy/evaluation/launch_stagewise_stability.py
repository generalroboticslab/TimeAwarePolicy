#!/usr/bin/env python3
"""Launch paired stagewise-stability evaluations in detached tmux jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

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
    / "stagewise_stability_evaluation.json"
)
PYTHON = Path(sys.executable)
TRAIN_RES = ROOT / "train_res"
EVAL_RES = ROOT / "eval_res"
OUT = ROOT / "results" / "final_three_task"
CAMPAIGN = OUT / "stagewise_stability" / "evaluation_campaign"
LOGS = CAMPAIGN / "logs"


CAMPAIGN_SUFFIX = "fullstable_repeat2"
FIXED_CONFIG_REPEATS = 2
SOURCE_BANK_SIZE = 1000
CANARY_NUM_ENVS = 128
FULL_NUM_ENVS = 2000
CHECKPOINT_INDEX = "best_rew"
SEED = 123456


def campaign_id(mode: str) -> str:
    return f"{mode}_{CAMPAIGN_SUFFIX}"


def manifest_path(mode: str) -> Path:
    return CAMPAIGN / f"{campaign_id(mode)}_manifest.json"


def status_path(mode: str) -> Path:
    return CAMPAIGN / f"{campaign_id(mode)}_status.json"

PROFILES = {}


def configure_profile(profile: dict) -> None:
    """Activate one validated stagewise-stability artifact profile."""
    global CAMPAIGN_SUFFIX, FIXED_CONFIG_REPEATS, SOURCE_BANK_SIZE
    global CANARY_NUM_ENVS, FULL_NUM_ENVS, CHECKPOINT_INDEX, SEED, PROFILES
    CAMPAIGN_SUFFIX = str(profile["campaign_suffix"])
    FIXED_CONFIG_REPEATS = int(profile["fixed_config_repeats"])
    SOURCE_BANK_SIZE = int(profile["source_bank_size"])
    CANARY_NUM_ENVS = int(profile["canary_num_envs"])
    FULL_NUM_ENVS = int(profile["full_num_envs"])
    CHECKPOINT_INDEX = str(profile["checkpoint_index"])
    SEED = int(profile["seed"])
    PROFILES = require_mapping(profile, "tasks", "profile")
    if min(
        FIXED_CONFIG_REPEATS,
        SOURCE_BANK_SIZE,
        CANARY_NUM_ENVS,
        FULL_NUM_ENVS,
    ) <= 0:
        raise ValueError("Stagewise-stability counts must all be positive")
    for label, count in (
        ("canary_num_envs", CANARY_NUM_ENVS),
        ("full_num_envs", FULL_NUM_ENVS),
    ):
        if count % FIXED_CONFIG_REPEATS:
            raise ValueError(f"{label} must be divisible by fixed_config_repeats")
    if FULL_NUM_ENVS != SOURCE_BANK_SIZE * FIXED_CONFIG_REPEATS:
        raise ValueError(
            "full_num_envs must equal source_bank_size * fixed_config_repeats"
        )


configure_profile(load_profile(DEFAULT_PROFILE, "stagewise_stability_evaluation"))


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def command_for(task_key: str, controller: str, gpu: int, num_envs: int,
                canary: bool = False) -> dict:
    profile = PROFILES[task_key]
    checkpoint = require_mapping(
        profile, "checkpoint", f"profile.tasks.{task_key}"
    ).get("directory")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(
            f"profile.tasks.{task_key}.checkpoint.directory must be a string"
        )
    evaluation_tag = f"stagewise_{'canary' if canary else 'full'}_{CAMPAIGN_SUFFIX}"
    command = [
        str(PYTHON), "-m", "projects.TimeAwarePolicy.eval",
        "--saving",
        "--task_name", profile["task"],
        "--train_res_dir", str(TRAIN_RES),
        "--eval_res_dir", str(EVAL_RES),
        "--checkpoint", checkpoint,
        "--index_episode", CHECKPOINT_INDEX,
        "--num_envs", str(num_envs),
        "--target_episodes", str(num_envs),
        "--target_success_eps", str(num_envs),
        "--target_record_eps", str(num_envs),
        "--save_threshold", "1",
        "--warmup_episodes", "0",
        "--episodeLength_eval", str(profile["episode_length"]),
        "--goal_speed", str(profile["goal_speed"]),
        "--budget_portion", json.dumps(profile["budget_portion"]),
        "--speed_describe", json.dumps(profile["speed_describe"]),
        "--fixed_config_repeats_eval", str(FIXED_CONFIG_REPEATS),
        "--eval_tag", evaluation_tag,
        "--strict_eval",
        "--record_init_configs",
        "--fixed_configs_eval", "true",
        "--par_configs_eval",
        "--paired_stage_eval",
        "--apply_noise_eval", "true",
        "--seed", str(SEED),
        "--sim_device", "cuda:0",
        "--graphics_device_id", "-1",
    ]
    if controller == "constant":
        command.append("--use_avg_speed")
    return {
        "id": (
            f"stagewise_{task_key}_{controller}"
            + ("_canary" if canary else "")
            + f"_{CAMPAIGN_SUFFIX}"
        ),
        "task_key": task_key,
        "task": profile["task"],
        "controller": controller,
        "gpu": gpu,
        "num_envs": num_envs,
        "goal_speed": profile["goal_speed"],
        "real_time_goal": "2 * per-configuration T_min",
        "source_bank_size": SOURCE_BANK_SIZE,
        "checkpoint": checkpoint,
        "fixed_config_repeats": FIXED_CONFIG_REPEATS,
        "unique_fixed_configs": num_envs // FIXED_CONFIG_REPEATS,
        "stable_stage_interval": "all executed stable-labelled steps",
        "command": command,
        "log": str(LOGS / (
            f"{task_key}_{controller}"
            + ("_canary" if canary else "")
            + f"_{CAMPAIGN_SUFFIX}"
            + ".log"
        )),
        "status": "pending",
        "output": None,
        "return_code": None,
    }


def jobs(mode: str, gpu_ids: tuple[int, ...] | None = None) -> list[dict]:
    gpu_ids = tuple(range(6)) if gpu_ids is None else gpu_ids
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    if mode == "canary":
        output = []
        job_index = 0
        for task_key in PROFILES:
            for controller in ("stage_wise", "constant"):
                output.append(
                    command_for(
                        task_key,
                        controller,
                        gpu_ids[job_index % len(gpu_ids)],
                        CANARY_NUM_ENVS,
                        canary=True,
                    )
                )
                job_index += 1
        return output
    output = []
    job_index = 0
    for task_key in PROFILES:
        for controller in ("stage_wise", "constant"):
            output.append(command_for(
                task_key,
                controller,
                gpu_ids[job_index % len(gpu_ids)],
                FULL_NUM_ENVS,
            ))
            job_index += 1
    return output


def validate_sources(job_list: list[dict]) -> None:
    for job in job_list:
        task_root = TRAIN_RES / job["task"]
        checkpoint = resolve_artifact(
            task_root,
            require_mapping(
                PROFILES[job["task_key"]],
                "checkpoint",
                f"profile.tasks.{job['task_key']}",
            ),
            f"stagewise.{job['task_key']}",
        )
        if checkpoint.name != job["checkpoint"]:
            raise RuntimeError(f"Resolved checkpoint changed for {job['id']}")
        policy = checkpoint / "checkpoints" / f"eps_{CHECKPOINT_INDEX}"
        if not policy.is_file():
            raise FileNotFoundError(policy)
        config = json.loads((checkpoint / "config.json").read_text())
        parent = config["checkpoint"]
        parent_index = config["index_episode"]
        bank = EVAL_RES / job["task"] / f"{parent}_EVAL_{parent_index}" / "trajectories" / "init_configs.json"
        if not bank.is_file():
            raise FileNotFoundError(bank)
        bank_payload = json.loads(bank.read_text())
        bank_sizes = {
            len(values) for values in bank_payload.values()
            if isinstance(values, list)
        }
        if bank_sizes != {SOURCE_BANK_SIZE}:
            raise RuntimeError(
                f"{bank} is not a consistent frozen {SOURCE_BANK_SIZE}-configuration bank: "
                f"{sorted(bank_sizes)}"
            )


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def launch(mode: str, gpu_ids: tuple[int, ...], profile_path: Path) -> None:
    job_list = jobs(mode, gpu_ids)
    validate_sources(job_list)
    LOGS.mkdir(parents=True, exist_ok=True)
    gpu_snapshot = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    current_campaign_id = campaign_id(mode)
    session = f"stagewise_motion_{current_campaign_id}"
    manifest = {
        "created_at": timestamp(),
        "mode": mode,
        "campaign_id": current_campaign_id,
        "stable_stage_interval": "all executed stable-labelled steps",
        "source_bank_size": SOURCE_BANK_SIZE,
        "fixed_config_repeats": FIXED_CONFIG_REPEATS,
        "purpose": (
            f"Paired strict {mode} evaluation of "
            f"{job_list[0]['num_envs']} rollouts per controller. Each of "
            f"{job_list[0]['unique_fixed_configs']} fixed bank configurations "
            f"is enumerated exactly {FIXED_CONFIG_REPEATS} times, identically "
            "for stage-wise and constant controllers. Both the time-average "
            "and peak object-motion proxy are measured over all executed "
            "stable-labelled steps. Each real-time goal is 2*T_min for its "
            "configuration."
        ),
        "checkpoint_selection": (
            "Successful N-P3O checkpoints selected in the final curve package, "
            "evaluated on their parent fixed configuration banks; "
            "these current-runtime checkpoints replace the observation-layout-"
            "incompatible archival stability-analysis checkpoints."
        ),
        "cwd": str(ROOT),
        "train_res_dir": str(TRAIN_RES),
        "eval_res_dir": str(EVAL_RES),
        "output_dir": str(OUT),
        "profile": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "python": str(PYTHON),
        "python_version": subprocess.run(
            [str(PYTHON), "--version"], text=True, capture_output=True, check=True
        ).stdout.strip(),
        "git_branch": git_output("branch", "--show-current"),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "gpu_snapshot": gpu_snapshot,
        "tmux_session": session,
        "jobs": job_list,
    }
    current_manifest = manifest_path(mode)
    current_status = status_path(mode)
    manifest["status_path"] = str(current_status)
    write_json(current_manifest, manifest)
    write_json(current_status, {"updated_at": timestamp(), "mode": mode, "jobs": job_list})
    worker = shlex.join([
        str(PYTHON), str(Path(__file__).resolve()),
        "--worker", str(current_manifest),
    ])
    shell_command = f"cd {shlex.quote(str(ROOT))} && {worker}"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, shell_command],
        check=True,
    )
    print(json.dumps({
        "session": session,
        "manifest": str(current_manifest),
        "status": str(current_status),
    }, indent=2))


def output_from_log(job: dict) -> Path | None:
    log = Path(job["log"])
    if not log.is_file():
        return None
    name = None
    for line in log.read_text(errors="replace").replace("\r", "\n").splitlines():
        if line.startswith("Uniform name is:"):
            name = line.split(":", 1)[1].strip()
    if not name:
        return None
    command = job["command"]
    eval_root = Path(command[command.index("--eval_res_dir") + 1])
    return eval_root / job["task"] / name


def validate_output(job: dict, output: Path) -> None:
    paired = output / "trajectories" / "paired_stage_metrics.json"
    metadata = output / "trajectories" / "meta_data.json"
    if not paired.is_file() or not metadata.is_file():
        raise RuntimeError(f"Missing final stagewise-stability artifacts in {output}")
    payload = json.loads(paired.read_text())
    repeats = int(job["fixed_config_repeats"])
    if payload.get("fixed_config_repeats_eval") != repeats:
        raise RuntimeError(f"Wrong fixed-config repeat count in {job['id']}")
    records = payload["records"]
    if len(records) != job["num_envs"]:
        raise RuntimeError(
            f"{job['id']} recorded {len(records)} rows, expected {job['num_envs']}"
        )
    source_indices = [record["source_config_index"] for record in records]
    expected_indices = list(range(job["unique_fixed_configs"]))
    observed_counts = {
        index: source_indices.count(index) for index in set(source_indices)
    }
    if sorted(observed_counts) != expected_indices or any(
        count != repeats for count in observed_counts.values()
    ):
        raise RuntimeError(
            f"{job['id']} did not enumerate each fixed configuration exactly "
            f"{repeats} times"
        )
    required_metrics = {
        "stable_object_motion_mean", "stable_object_motion_peak",
        "stable_object_motion_sum", "stable_stage_steps",
    }
    for record in records:
        missing = required_metrics.difference(record)
        if missing:
            raise RuntimeError(
                f"{job['id']} record is missing metrics: {sorted(missing)}"
            )


def worker(current_manifest: Path) -> None:
    manifest = read_json(current_manifest)
    current_status = Path(manifest.get(
        "status_path",
        status_path(manifest["mode"]),
    ))
    job_list = manifest["jobs"]
    running = []
    for job in job_list:
        log_path = Path(job["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("w")
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
            "PATH": ":".join(filter(None, [
                str(PYTHON.parent),
                env.get("PATH", ""),
            ])),
            "PYTHONPATH": ":".join([
                str(ROOT / "isaacgym" / "python"),
                str(ROOT / "envs"),
                str(ROOT),
            ]),
            "LD_LIBRARY_PATH": ":".join(filter(None, [
                str(PYTHON.parent.parent / "lib"),
                env.get("LD_LIBRARY_PATH", ""),
            ])),
            "WANDB_MODE": "offline",
            "PYTHONUNBUFFERED": "1",
        })
        process = subprocess.Popen(
            job["command"], cwd=ROOT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, text=True,
        )
        job["status"] = "running"
        job["pid"] = process.pid
        running.append((job, process, stream))
    write_json(current_status, {"updated_at": timestamp(), "mode": manifest["mode"], "jobs": job_list})

    while running:
        remaining = []
        for job, process, stream in running:
            return_code = process.poll()
            if return_code is None:
                remaining.append((job, process, stream))
                continue
            stream.close()
            job["return_code"] = return_code
            output = output_from_log(job)
            job["output"] = str(output) if output else None
            try:
                if return_code != 0:
                    raise RuntimeError(f"process returned {return_code}")
                if output is None:
                    raise RuntimeError("output name was not printed")
                validate_output(job, output)
                job["status"] = "completed"
            except Exception as error:
                job["status"] = "failed"
                job["error"] = str(error)
        running = remaining
        write_json(current_status, {"updated_at": timestamp(), "mode": manifest["mode"], "jobs": job_list})
        if running:
            time.sleep(10)


def main() -> None:
    global OUT, CAMPAIGN, LOGS, TRAIN_RES, EVAL_RES
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", choices=("canary", "full"))
    parser.add_argument("--worker", type=Path)
    parser.add_argument(
        "--profile", type=Path, default=DEFAULT_PROFILE,
        help="Versioned stagewise-stability checkpoint and schedule profile.",
    )
    parser.add_argument("--train-res-dir", type=Path, default=TRAIN_RES)
    parser.add_argument("--eval-res-dir", type=Path, default=EVAL_RES)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5",
        help="Comma-separated CUDA device indices.",
    )
    parser.add_argument(
        "--preflight-only",
        choices=("canary", "full"),
        help="Resolve and validate all sources without launching tmux.",
    )
    args = parser.parse_args()
    profile_path = args.profile.resolve()
    configure_profile(load_profile(profile_path, "stagewise_stability_evaluation"))
    TRAIN_RES = args.train_res_dir.resolve()
    EVAL_RES = args.eval_res_dir.resolve()
    OUT = args.output_dir.resolve()
    CAMPAIGN = OUT / "stagewise_stability" / "evaluation_campaign"
    LOGS = CAMPAIGN / "logs"
    gpu_ids = tuple(int(item) for item in args.gpus.split(",") if item.strip())
    if not gpu_ids:
        parser.error("--gpus must contain at least one device index")
    if args.worker:
        worker(args.worker)
    elif args.preflight_only:
        job_list = jobs(args.preflight_only, gpu_ids)
        validate_sources(job_list)
        print(json.dumps({
            "status": "passed",
            "profile": str(profile_path),
            "profile_sha256": sha256_file(profile_path),
            "jobs": [
                {key: job[key] for key in (
                    "id", "task", "controller", "checkpoint", "num_envs"
                )}
                for job in job_list
            ],
        }, indent=2))
    elif args.launch:
        launch(args.launch, gpu_ids, profile_path)
    else:
        parser.error("choose --launch, --preflight-only, or --worker")


if __name__ == "__main__":
    main()
