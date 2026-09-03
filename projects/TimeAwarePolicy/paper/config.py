"""Validated configuration loading for public result-reproduction scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = 1


def load_profile(path: Path, expected_kind: str) -> dict:
    """Load a versioned JSON profile and validate its public schema header."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON profile {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Profile {path} must contain a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Profile {path} uses schema_version={payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if payload.get("profile_kind") != expected_kind:
        raise ValueError(
            f"Profile {path} has profile_kind={payload.get('profile_kind')!r}; "
            f"expected {expected_kind!r}"
        )
    return payload


def require_mapping(mapping: dict, key: str, context: str) -> dict:
    """Return a required mapping value with an actionable schema error."""
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be a JSON object")
    return value


def require_sequence(mapping: dict, key: str, context: str) -> list:
    """Return a required non-empty list with an actionable schema error."""
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty JSON array")
    return value


def resolve_artifact(task_root: Path, specification: dict, context: str) -> Path:
    """Resolve one exact directory or one unambiguous glob from a profile."""
    directory = specification.get("directory")
    pattern = specification.get("pattern")
    if bool(directory) == bool(pattern):
        raise ValueError(
            f"{context} must define exactly one of 'directory' or 'pattern'"
        )
    if directory:
        path = task_root / str(directory)
        if not path.is_dir():
            raise FileNotFoundError(
                f"Missing artifact for {context}: {path}. "
                "Install the optional full-result artifact bundle or override "
                "--train-res-dir."
            )
        return path
    matches = sorted(task_root.glob(str(pattern)))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one artifact for {context} matching "
            f"{task_root / str(pattern)}, found {matches}"
        )
    return matches[0]


def resolve_root_path(root: Path, value: str) -> Path:
    """Resolve a profile path relative to the repository unless absolute."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a profile or provenance input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
