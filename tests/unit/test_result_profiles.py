import json
from pathlib import Path

import pytest

from projects.TimeAwarePolicy.paper.config import load_profile, resolve_artifact
from projects.TimeAwarePolicy.evaluation.launch_policy_evaluations import selected_jobs


ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = ROOT / "projects" / "TimeAwarePolicy" / "paper" / "configs"


@pytest.mark.parametrize(
    "name,kind",
    (
        ("policy_evaluation.json", "policy_evaluation"),
        (
            "stagewise_stability_evaluation.json",
            "stagewise_stability_evaluation",
        ),
        ("figure_generation.json", "figure_generation"),
    ),
)
def test_public_result_profiles_are_versioned(name, kind):
    profile = load_profile(PROFILE_ROOT / name, kind)
    assert profile["schema_version"] == 1
    assert profile["profile_kind"] == kind


def test_profile_loader_rejects_wrong_kind(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "profile_kind": "different",
    }))
    with pytest.raises(ValueError, match="profile_kind"):
        load_profile(path, "expected")


def test_artifact_resolver_requires_one_unambiguous_source(tmp_path):
    task_root = tmp_path / "Task"
    task_root.mkdir()
    (task_root / "checkpoint_a").mkdir()
    assert resolve_artifact(
        task_root, {"directory": "checkpoint_a"}, "test"
    ).name == "checkpoint_a"
    assert resolve_artifact(
        task_root, {"pattern": "checkpoint_*"}, "test"
    ).name == "checkpoint_a"
    with pytest.raises(ValueError, match="exactly one"):
        resolve_artifact(task_root, {}, "test")


def test_policy_jobs_are_derived_from_profile_roles(tmp_path):
    task_root = tmp_path / "Task"
    (task_root / "time_checkpoint").mkdir(parents=True)
    (task_root / "cmdp_checkpoint").mkdir()
    profile = {
        "tasks": {
            "example": {
                "task": "Task",
                "time_optimal": {
                    "Q40": {"directory": "time_checkpoint"},
                },
                "cmdp": {
                    "N-P3O": {"directory": "cmdp_checkpoint"},
                },
            }
        }
    }
    jobs = selected_jobs(profile, tmp_path)
    assert [(job["group"], job["comparison"]) for job in jobs] == [
        ("time_optimal", "Q40"),
        ("cmdp", "N-P3O"),
    ]


def test_dated_artifact_ids_live_in_profiles_not_executable_code():
    source_roots = (
        ROOT / "core",
        ROOT / "projects" / "TimeAwarePolicy",
        ROOT / "exec" / "TimeAwarePolicy",
    )
    sources = sorted(
        path
        for source_root in source_roots
        for path in source_root.rglob("*")
        if path.suffix in {".py", ".sh"}
    )
    assert sources
    for path in sources:
        source = path.read_text()
        assert "202608" not in source, path
        assert "full3_2026" not in source, path
        assert "additive15_2026" not in source, path

    policy_profile = (PROFILE_ROOT / "policy_evaluation.json").read_text()
    stagewise_profile = (
        PROFILE_ROOT / "stagewise_stability_evaluation.json"
    ).read_text()
    assert "202608" in policy_profile
    assert "202608" in stagewise_profile


def test_stagewise_stability_uses_descriptive_public_names():
    profile = json.loads((PROFILE_ROOT / "figure_generation.json").read_text())
    assert "stagewise_stability_status" in profile
    assert "stagewise_stability_evaluation_profile" in profile

    legacy_token = "figure" + "8"
    public_sources = (
        ROOT / "projects" / "TimeAwarePolicy" / "evaluation"
        / "launch_stagewise_stability.py",
        ROOT / "projects" / "TimeAwarePolicy" / "paper" / "build_results.py",
        ROOT / "projects" / "TimeAwarePolicy" / "paper" / "reports.py",
        ROOT / "projects" / "TimeAwarePolicy" / "paper" / "figures"
        / "stagewise_stability.py",
    )
    for path in public_sources:
        assert legacy_token not in path.read_text().lower(), path
