#!/usr/bin/env python3
"""Run one end-to-end training update for each public simulator task."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys


TASKS = ("FrankaCubeStack", "FrankaGmPour", "FrankaCabinet")
SUCCESS_MARKER = "Process Over here"


def git_output(repository, *args):
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def training_command(repository, task, num_envs, num_steps, output_dir):
    return [
        sys.executable,
        "-m",
        "projects.TimeAwarePolicy.train",
        "--task_name", task,
        "--num_envs", str(num_envs),
        "--num-steps", str(num_steps),
        "--minibatch-size", str(num_envs * num_steps),
        "--update_epochs", "1",
        "--num_updates", "1",
        "--nographics",
        "--graphics_device_id", "-1",
        "--saving", "false",
        "--wandb", "false",
        "--quiet", "false",
        "--train_res_dir", str(output_dir / "scratch_train"),
        "--eval_res_dir", str(output_dir / "scratch_eval"),
    ]


def runtime_environment(repository, gpu):
    """Build a self-contained environment for the selected Python runtime."""
    environment = os.environ.copy()
    environment_bin = Path(sys.executable).resolve().parent
    environment_lib = environment_bin.parent / "lib"
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONDONTWRITEBYTECODE": "1",
        "WANDB_MODE": "disabled",
        "PATH": os.pathsep.join(filter(None, (
            str(environment_bin), environment.get("PATH"),
        ))),
        "LD_LIBRARY_PATH": os.pathsep.join(filter(None, (
            str(environment_lib), environment.get("LD_LIBRARY_PATH"),
        ))),
        "PYTHONPATH": os.pathsep.join([
            str(repository / "isaacgym" / "python"),
            str(repository / "envs"),
            str(repository),
        ]),
    })
    return environment


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--expected-head")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args(argv)
    if min(args.num_envs, args.num_steps, args.timeout_seconds) <= 0:
        parser.error("environment count, rollout steps, and timeout must be positive")

    repository = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    head_before = git_output(repository, "rev-parse", "HEAD")
    status_before = git_output(repository, "status", "--porcelain")
    if args.expected_head and head_before != args.expected_head:
        parser.error(f"release HEAD is {head_before}, expected {args.expected_head}")
    if status_before:
        parser.error("release worktree must be clean before GPU smoke tests")

    environment = runtime_environment(repository, args.gpu)
    records = []
    for task in args.tasks:
        command = training_command(
            repository, task, args.num_envs, args.num_steps, output_dir
        )
        log_path = output_dir / f"{task}.log"
        started_at = dt.datetime.now().astimezone()
        with log_path.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=str(repository),
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
            )
        finished_at = dt.datetime.now().astimezone()
        output = log_path.read_text(errors="replace")
        records.append({
            "task": task,
            "command": command,
            "log": str(log_path),
            "return_code": result.returncode,
            "success_marker_found": SUCCESS_MARKER in output,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        })
        atomic_json(output_dir / "manifest.json", {
            "result": "running",
            "repository": str(repository),
            "git_head": head_before,
            "gpu": args.gpu,
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "records": records,
        })
        if result.returncode != 0 or SUCCESS_MARKER not in output:
            break

    head_after = git_output(repository, "rev-parse", "HEAD")
    status_after = git_output(repository, "status", "--porcelain")
    succeeded = (
        len(records) == len(args.tasks)
        and all(
            record["return_code"] == 0 and record["success_marker_found"]
            for record in records
        )
        and head_after == head_before
        and not status_after
    )
    atomic_json(output_dir / "manifest.json", {
        "result": "complete" if succeeded else "failed",
        "repository": str(repository),
        "git_head_before": head_before,
        "git_head_after": head_after,
        "git_status_after": status_after,
        "gpu": args.gpu,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "records": records,
    })
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
