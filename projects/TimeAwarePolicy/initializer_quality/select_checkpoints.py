"""Evaluate unlabeled policy candidates on held-out full domain randomization."""

import argparse
import csv
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGETS = [0.4, 0.6, 0.8, 0.95]


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_candidates(producer_dir):
    producer_dir = Path(producer_dir)
    checkpoint_dir = producer_dir / "checkpoints"
    metadata = read_json(producer_dir / "trajectories" / "meta_data.json")
    declared = metadata.get("quality_candidates", {})
    candidates = []
    for name, record in declared.items():
        if not name.startswith("candidate_u"):
            continue
        policy = checkpoint_dir / f"eps_{name}"
        normalizer = checkpoint_dir / f"rew_norm_eps_{name}"
        if not policy.is_file() or not normalizer.is_file():
            raise FileNotFoundError(f"candidate {name} is missing its policy or reward normalizer")
        candidates.append({
            "name": name,
            "accepted_update": int(record["accepted_update"]),
            "attempted_update": int(record["attempted_update"]),
            "rolling_success_rate": float(record["rolling_success_rate"]),
            "curriculum_ratio": float(record["curriculum_ratio"]),
            "policy": policy,
            "normalizer": normalizer,
        })
    candidates.sort(key=lambda item: item["accepted_update"])
    if not candidates:
        raise ValueError(f"no declared quality candidates found in {producer_dir}")
    return candidates


def evaluation_output(eval_res_dir, task_name, producer_config, candidate_name):
    suffix = f"_EVAL_{candidate_name}"
    final_name = producer_config["final_name"]
    final_name = final_name[:max(250 - len(suffix), 0)] + suffix
    return Path(eval_res_dir) / task_name / final_name


def evaluation_command(args, candidate_name):
    return [
        sys.executable,
        "-m",
        "projects.TimeAwarePolicy.eval",
        "--saving",
        "--task_name", args.task_name,
        "--train_res_dir", str(args.train_res_dir.resolve()),
        "--eval_res_dir", str(args.eval_res_dir.resolve()),
        "--checkpoint", args.producer,
        "--index_episode", candidate_name,
        "--num_envs", str(args.num_envs),
        "--target_success_eps", str(args.num_envs),
        "--strict_eval",
        "--fixed_configs_eval", "false",
        "--init_curri_ratio", "1",
        "--apply_noise_eval", "true",
        "--deterministic", "true",
        "--seed", str(args.seed),
        "--sim_device", args.sim_device,
        "--graphics_device_id", str(args.graphics_device_id),
    ]


def read_strict_evaluation(output, expected):
    output = Path(output)
    config = read_json(output / "config.json")
    metadata = read_json(output / "trajectories" / "meta_data.json")
    with (output / "data.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"expected one aggregate row in {output / 'data.csv'}")
    checks = {
        "strict_eval": config.get("strict_eval") is True,
        "num_envs": config.get("num_envs") == expected["num_envs"],
        "target_success_eps": config.get("target_success_eps") == expected["num_envs"],
        "seed": config.get("seed") == expected["seed"],
        "init_curri_ratio": config.get("init_curri_ratio") == 1,
        "fixed_configs": config.get("fixed_configs") is False,
        "apply_noise_eval": config.get("apply_noise_eval") is True,
        "episodes": metadata.get("episode") == expected["num_envs"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid held-out evaluation {output}: {', '.join(failed)}")
    success_rate = float(rows[0]["success_rate"])
    if not 0 <= success_rate <= 1:
        raise ValueError(f"invalid success rate {success_rate} in {output}")
    return success_rate


def assign_distinct(evaluations, targets):
    """Minimize total target error while using every candidate at most once."""
    target_count = len(targets)
    states = {0: (0.0, [None] * target_count)}
    for candidate_index, evaluation in enumerate(evaluations):
        updated = dict(states)
        for mask, (cost, assignment) in states.items():
            for target_index, target in enumerate(targets):
                bit = 1 << target_index
                if mask & bit:
                    continue
                next_mask = mask | bit
                next_cost = cost + abs(evaluation["success_rate"] - target)
                incumbent = updated.get(next_mask)
                if incumbent is None or next_cost < incumbent[0]:
                    next_assignment = list(assignment)
                    next_assignment[target_index] = candidate_index
                    updated[next_mask] = (next_cost, next_assignment)
        states = updated
    full_mask = (1 << target_count) - 1
    if full_mask not in states:
        raise ValueError(f"need at least {target_count} distinct evaluated candidates")
    return [evaluations[index] for index in states[full_mask][1]]


def seal_quality_bank(producer_dir, output_dir, evaluations, targets, protocol):
    producer_dir = Path(producer_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite quality bank: {output_dir}")
    selected = assign_distinct(evaluations, targets)
    (output_dir / "checkpoints").mkdir(parents=True)
    (output_dir / "trajectories").mkdir()
    shutil.copy2(producer_dir / "config.json", output_dir / "config.json")

    milestone = {}
    selection_records = {}
    for target, evaluation in zip(targets, selected):
        label = f"quality_{round(target * 100):02d}"
        target_policy = output_dir / "checkpoints" / f"eps_{label}"
        target_normalizer = output_dir / "checkpoints" / f"rew_norm_eps_{label}"
        shutil.copy2(evaluation["policy"], target_policy)
        shutil.copy2(evaluation["normalizer"], target_normalizer)
        record = {
            "target_success_rate": target,
            "success_rate": evaluation["success_rate"],
            "success_percentage": 100 * evaluation["success_rate"],
            "numerator": round(evaluation["success_rate"] * protocol["num_envs"]),
            "denominator": protocol["num_envs"],
            "absolute_error": abs(evaluation["success_rate"] - target),
            "source_index": evaluation["name"],
            "source_accepted_update": evaluation["accepted_update"],
            "source_attempted_update": evaluation["attempted_update"],
            "source_curriculum_ratio": evaluation["curriculum_ratio"],
            "source": "held_out_strict_full_distribution_evaluation",
            "evaluation_output": str(evaluation["output"]),
            "policy_sha256": sha256(target_policy),
            "reward_normalizer_sha256": sha256(target_normalizer),
        }
        milestone[label] = record
        selection_records[label] = record

    metadata = {
        "milestone": milestone,
        "quality_targets_observed": {f"{round(value * 100):02d}": True for value in targets},
        "quality_targets_complete": True,
        "training_info": {},
        "calibration_protocol": protocol,
    }
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": str(producer_dir.resolve()),
        "protocol": protocol,
        "evaluated_candidates": [
            {
                "name": item["name"],
                "accepted_update": item["accepted_update"],
                "attempted_update": item["attempted_update"],
                "training_curriculum_ratio": item["curriculum_ratio"],
                "training_rolling_success_rate": item["rolling_success_rate"],
                "held_out_success_rate": item["success_rate"],
                "evaluation_output": str(item["output"]),
            }
            for item in evaluations
        ],
        "selections": selection_records,
    }
    write_json(output_dir / "trajectories" / "meta_data.json", metadata)
    write_json(output_dir / "trajectories" / "calibration_provenance.json", provenance)
    return selected


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--producer", required=True, help="Training-result folder name.")
    parser.add_argument("--output_bank", required=True, help="New calibrated bank folder name.")
    parser.add_argument("--train_res_dir", type=Path, default=Path("train_res"))
    parser.add_argument("--eval_res_dir", type=Path, default=Path("eval_res"))
    parser.add_argument("--num_envs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--targets", type=json.loads, default=DEFAULT_TARGETS)
    parser.add_argument("--stop_success", type=float, default=0.95)
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=-1)
    parser.add_argument(
        "--execute", action="store_true",
        help="Run missing evaluations and seal the bank; otherwise print commands only.",
    )
    args = parser.parse_args(argv)
    args.train_res_dir = args.train_res_dir.resolve()
    args.eval_res_dir = args.eval_res_dir.resolve()
    if args.num_envs <= 0:
        parser.error("--num_envs must be positive")
    if len(args.targets) != 4 or len(set(args.targets)) != 4:
        parser.error("--targets must contain four distinct success rates")
    if any(not 0 <= target <= 1 for target in args.targets):
        parser.error("every target must be in [0, 1]")
    if not 0 <= args.stop_success <= 1:
        parser.error("--stop_success must be in [0, 1]")
    return args


def main(argv=None):
    args = parse_args(argv)
    producer_dir = args.train_res_dir / args.task_name / args.producer
    output_dir = args.train_res_dir / args.task_name / args.output_bank
    producer_config = read_json(producer_dir / "config.json")
    if producer_config.get("task_name") != args.task_name:
        raise ValueError("--task_name does not match the producer configuration")
    if args.execute and output_dir.exists():
        raise FileExistsError(f"refusing to overwrite quality bank: {output_dir}")
    candidates = discover_candidates(producer_dir)
    evaluations = []
    expected = {"num_envs": args.num_envs, "seed": args.seed}

    for candidate in candidates:
        command = evaluation_command(args, candidate["name"])
        output = evaluation_output(args.eval_res_dir, args.task_name, producer_config, candidate["name"])
        if not output.is_dir():
            if not args.execute:
                print(shlex.join(command))
                continue
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        success_rate = read_strict_evaluation(output, expected)
        evaluation = dict(candidate, success_rate=success_rate, output=output)
        evaluations.append(evaluation)
        print(f"{candidate['name']}: held-out success {100 * success_rate:.2f}%")
        if len(evaluations) >= len(args.targets) and success_rate >= args.stop_success:
            break

    if not args.execute:
        print("Dry run only. Re-run with --execute to evaluate and seal the bank.")
        return 0

    protocol = {
        "strict_eval": True,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "init_curri_ratio": 1.0,
        "fixed_configs": False,
        "apply_noise_eval": True,
        "candidate_interval_accepted_updates": producer_config["quality_candidate_interval"],
        "candidate_start_rolling_success": producer_config["quality_candidate_start_success"],
        "selection_rule": "minimum total absolute target error with distinct candidates",
        "targets": args.targets,
    }
    selected = seal_quality_bank(producer_dir, output_dir, evaluations, args.targets, protocol)
    for target, item in zip(args.targets, selected):
        print(f"quality_{round(target * 100):02d} <- {item['name']} ({100 * item['success_rate']:.2f}%)")
    print(f"Calibrated bank: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
