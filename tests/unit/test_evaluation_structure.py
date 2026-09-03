import ast
import subprocess
import sys
from pathlib import Path

import numpy as np

from projects.TimeAwarePolicy.evaluation.scene_visualization import quat_to_rot


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_identity_quaternion_has_identity_rotation():
    np.testing.assert_allclose(quat_to_rot([0, 0, 0, 1]), np.eye(3))


def test_argument_module_does_not_eagerly_import_matplotlib():
    command = [
        sys.executable,
        "-c",
        (
            "import sys; import projects.TimeAwarePolicy.arguments.evaluation; "
            "assert 'matplotlib' not in sys.modules"
        ),
    ]
    subprocess.run(command, check=True)


def test_real_robot_evaluation_methods_live_in_protected_package_once():
    method_names = {"evaluate_real_robot", "_save_debug_data"}
    paths = (
        REPOSITORY_ROOT / "core" / "evaluation" / "evaluator.py",
        REPOSITORY_ROOT / "real_robot" / "evaluation.py",
    )
    definitions = []
    for path in paths:
        tree = ast.parse(path.read_text())
        definitions.extend(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in method_names
        )
    assert sorted(definitions) == sorted(method_names)


def test_stagewise_stability_methods_live_in_project_package_once():
    method_names = {
        "initialize_paired_stage_metrics",
        "_record_paired_stage_metrics",
        "save_paired_stage_metrics",
    }
    paths = (
        REPOSITORY_ROOT / "core" / "evaluation" / "evaluator.py",
        REPOSITORY_ROOT
        / "projects"
        / "TimeAwarePolicy"
        / "evaluation"
        / "stagewise_stability.py",
    )
    definitions = []
    for path in paths:
        tree = ast.parse(path.read_text())
        definitions.extend(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in method_names
        )
    assert sorted(definitions) == sorted(method_names)


def test_episode_metrics_live_in_metrics_module_once():
    method_names = {
        "update_episode_metrics",
        "compute_average_metrics",
        "update_speed_time_dict",
        "save_results",
        "_save_init_configs",
    }
    paths = (
        REPOSITORY_ROOT / "core" / "evaluation" / "evaluator.py",
        REPOSITORY_ROOT / "core" / "evaluation" / "metrics.py",
    )
    definitions = []
    for path in paths:
        tree = ast.parse(path.read_text())
        definitions.extend(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in method_names
        )
    assert sorted(definitions) == sorted(method_names)
