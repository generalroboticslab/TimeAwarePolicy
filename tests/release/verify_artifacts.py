#!/usr/bin/env python3
"""Verify the bundled public checkpoints and temporal calibration banks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    ROOT
    / "projects"
    / "TimeAwarePolicy"
    / "paper"
    / "configs"
    / "bundled_files.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported artifact manifest schema")
    if manifest.get("profile_kind") != "bundled_files":
        raise ValueError("Not a bundled-file manifest")

    checked = []
    for task, task_spec in manifest["tasks"].items():
        for role in ("initializer", "time_aware_policy", "reference_bank"):
            artifact = task_spec[role]
            directory = ROOT / artifact["directory"]
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            for relative, expected_hash in artifact["files"].items():
                path = directory / relative
                if not path.is_file():
                    raise FileNotFoundError(path)
                actual_hash = sha256_file(path)
                if actual_hash != expected_hash:
                    raise RuntimeError(
                        f"SHA-256 mismatch for {path}: "
                        f"expected {expected_hash}, found {actual_hash}"
                    )
                checked.append(str(path.relative_to(ROOT)))

        bank = task_spec["reference_bank"]
        bank_path = ROOT / bank["directory"] / "trajectories/init_configs.json"
        bank_data = json.loads(bank_path.read_text())
        expected_count = int(bank["configurations"])
        sequence_lengths = {
            key: len(value)
            for key, value in bank_data.items()
            if isinstance(value, list)
        }
        if not sequence_lengths or set(sequence_lengths.values()) != {expected_count}:
            raise RuntimeError(
                f"Calibration-bank cardinality mismatch for {task}: "
                f"{sequence_lengths}"
            )

    return {
        "status": "passed",
        "manifest": str(manifest_path),
        "tasks": len(manifest["tasks"]),
        "files_checked": len(checked),
        "checked_files": checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest.resolve()), indent=2))


if __name__ == "__main__":
    main()
