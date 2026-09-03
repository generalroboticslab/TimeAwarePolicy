"""Structural guarantees for the transparent public shell entrypoints."""

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
EXEC_ROOT = ROOT / "exec" / "TimeAwarePolicy"


def test_shell_entrypoints_are_executable_and_call_one_python_module():
    scripts = sorted(EXEC_ROOT.rglob("*.sh"))
    assert scripts
    for script in scripts:
        source = script.read_text()
        assert script.stat().st_mode & 0o111
        assert "python -m projects.TimeAwarePolicy." in source
        assert "subprocess" not in source
        assert ".py" not in source


def test_stage_recipes_expose_public_task_budgets_and_horizons():
    initial = (EXEC_ROOT / "train" / "initial_policy.sh").read_text()
    time_optimal = (EXEC_ROOT / "train" / "time_optimal.sh").read_text()
    time_aware = (EXEC_ROOT / "train" / "time_aware.sh").read_text()
    gmpour = (
        ROOT / "envs/isaacgymenvs/cfg/task/FrankaGmPour.yaml"
    ).read_text()

    for expected in ("updates=2500; horizon=500", "updates=6000; horizon=500"):
        assert expected in initial
    assert "updates=2500; horizon=800" in initial

    assert "updates=1500; horizon=2000" in time_aware
    assert "updates=2500; horizon=1600" in time_aware
    assert "updates=1500; horizon=2600" in time_aware
    assert "--successRewardScale 1000" in initial
    assert "--successRewardScale 1000" in time_aware
    assert "holdRewardScale: 5" in gmpour
    assert "--no_dense" not in initial
    assert "--no_dense" in time_optimal
    assert "--no_dense" in time_aware


def test_repository_root_has_no_python_entrypoint_clutter():
    assert not list(ROOT.glob("*.py"))


def test_readme_uses_named_shell_workflows():
    readme = (ROOT / "README.md").read_text()
    assert "python -m" not in readme
    assert 'initial_policy.sh "$TASK"' not in readme
    assert '--task "$TASK"' in readme


def test_test_workflow_wrappers_are_executable():
    scripts = (
        "tests/run_unit.sh",
        "tests/release/check.sh",
        "tests/release/smoke.sh",
    )
    for relative in scripts:
        script = ROOT / relative
        assert script.stat().st_mode & 0o111
        assert "python -m" in script.read_text()


def _captured_python_arguments(tmp_path, script, *arguments):
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    capture = tmp_path / "python-arguments.txt"
    python = executable_dir / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURE_FILE\"\n"
    )
    python.chmod(0o755)
    environment = os.environ.copy()
    environment["CAPTURE_FILE"] = str(capture)
    environment["PATH"] = f"{executable_dir}{os.pathsep}{environment['PATH']}"
    result = subprocess.run(
        ["bash", str(EXEC_ROOT / script), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return capture.read_text().splitlines()


def _assert_argument_pairs(arguments, *expected_pairs):
    for expected in expected_pairs:
        offset = arguments.index(expected[0])
        assert arguments[offset:offset + 2] == list(expected)


def test_initial_policy_wrapper_preserves_the_public_recipe(tmp_path):
    arguments = _captured_python_arguments(
        tmp_path,
        "train/initial_policy.sh",
        "--task", "FrankaGmPour",
    )
    for switch in ("--saving", "--fix_priv"):
        assert switch in arguments
    _assert_argument_pairs(
        arguments,
        ("--task_name", "FrankaGmPour"),
        ("--num_updates", "6000"),
        ("--episodeLength", "500"),
        ("--gamma", "0.995"),
        ("--value_bootstrap", "false"),
        ("--successRewardScale", "1000"),
        ("--quality_candidate_interval", "5"),
        ("--quality_candidate_start_success", "0.90"),
    )


def test_named_training_wrapper_arguments_are_explicit_and_forwarded(tmp_path):
    arguments = _captured_python_arguments(
        tmp_path,
        "train/time_optimal.sh",
        "--task", "FrankaCubeStack",
        "--checkpoint", "initializer_run",
        "--index", "quality_95",
        "--seed", "7",
    )
    assert arguments[:2] == ["-m", "projects.TimeAwarePolicy.train"]
    for expected in (
        ["--task_name", "FrankaCubeStack"],
        ["--checkpoint", "initializer_run"],
        ["--index_episode", "quality_95"],
        ["--num_updates", "1500"],
        ["--seed", "7"],
    ):
        offset = arguments.index(expected[0])
        assert arguments[offset:offset + 2] == expected
    for switch in ("--saving", "--fix_priv", "--reset_critic", "--no_dense"):
        assert switch in arguments
    _assert_argument_pairs(
        arguments,
        ("--warmup_iters", "50"),
        ("--episodeLength", "500"),
        ("--gamma", "0.995"),
        ("--value_bootstrap", "false"),
        ("--target_kl", "2.5"),
        ("--successRewardScale", "1000"),
        ("--epstimeRewardScale", "[100, 100]"),
    )


def test_initializer_quality_wrapper_hides_protocol_defaults(tmp_path):
    arguments = _captured_python_arguments(
        tmp_path,
        "train/initializer_quality.sh",
        "--task", "FrankaGmPour",
        "--producer", "initial_run",
        "--output-bank", "quality_bank",
        "--execute",
    )
    assert arguments[:2] == [
        "-m", "projects.TimeAwarePolicy.initializer_quality.select_checkpoints"
    ]
    assert arguments[-1] == "--execute"
    for expected in (
        ["--task_name", "FrankaGmPour"],
        ["--producer", "initial_run"],
        ["--output_bank", "quality_bank"],
        ["--num_envs", "2000"],
    ):
        offset = arguments.index(expected[0])
        assert arguments[offset:offset + 2] == expected


@pytest.mark.parametrize(
    ("task", "checkpoint"),
    (
        ("FrankaCubeStack", "20250717_162724_tw_FrankaCubeStack"),
        ("FrankaGmPour", "20250715_123940_tw_FrankaGmPour"),
        ("FrankaCabinet", "20250730_151924_tw_FrankaCabinet"),
    ),
)
def test_interactive_wrapper_preserves_the_public_demo_protocol(
    tmp_path, task, checkpoint
):
    arguments = _captured_python_arguments(
        tmp_path,
        "eval/interactive.sh",
        "--task", task,
        "--graphics_device_id", "2",
    )
    checkpoint_offset = arguments.index("--checkpoint")
    assert arguments[checkpoint_offset:checkpoint_offset + 2] == [
        "--checkpoint", checkpoint
    ]
    par_configs = arguments.index("--par_configs_eval")
    assert arguments[par_configs:par_configs + 2] == [
        "--par_configs_eval", "true"
    ]
    for switch in (
        "--rendering",
        "--keyboard_ctrl",
        "--simple_layout",
        "--draw_scevel",
    ):
        assert switch in arguments
    for expected in (
        ["--num_envs", "1"],
        ["--index_episode", "best_rew"],
        ["--goal_speed", "0.6"],
    ):
        offset = arguments.index(expected[0])
        assert arguments[offset:offset + 2] == expected
    assert arguments[-2:] == ["--graphics_device_id", "2"]


def test_temporal_student_explicitly_preserves_temporal_inputs(tmp_path):
    arguments = _captured_python_arguments(
        tmp_path,
        "train/temporal_student.sh",
        "--task", "FrankaCubeStack",
        "--checkpoint", "teacher_run",
    )
    for switch in ("--stu_train", "--warmup_rand", "--time2end", "--time_ratio"):
        assert switch in arguments
    _assert_argument_pairs(
        arguments,
        ("--task_name", "FrankaCubeStack"),
        ("--checkpoint", "teacher_run"),
        ("--index_episode", "best_rew"),
        ("--num_updates", "1500"),
        ("--episodeLength", "500"),
        ("--lr", "5e-4"),
        ("--gamma", "0.995"),
        ("--value_bootstrap", "false"),
        ("--wandb", "false"),
    )


def test_temporal_bank_wrapper_preserves_collection_protocol(tmp_path):
    arguments = _captured_python_arguments(
        tmp_path,
        "eval/temporal_bank.sh",
        "--checkpoint", "student_run",
    )
    for switch in ("--saving", "--record_init_configs", "--use_par_checkpoint"):
        assert switch in arguments
    _assert_argument_pairs(
        arguments,
        ("--graphics_device_id", "-1"),
        ("--num_envs", "10000"),
        ("--target_success_eps", "10000"),
        ("--target_record_eps", "1000"),
        ("--save_threshold", "10"),
        ("--checkpoint", "student_run"),
        ("--index_episode", "best"),
    )


def test_time_aware_wrapper_preserves_cmdp_protocol(tmp_path):
    arguments = _captured_python_arguments(
        tmp_path,
        "train/time_aware.sh",
        "--task", "FrankaCabinet",
        "--checkpoint", "student_run",
    )
    for switch in (
        "--saving",
        "--reset_critic",
        "--no_dense",
        "--time2end",
        "--time_ratio",
        "--fixed_configs",
        "--use_cost",
    ):
        assert switch in arguments
    _assert_argument_pairs(
        arguments,
        ("--task_name", "FrankaCabinet"),
        ("--checkpoint", "student_run"),
        ("--index_episode", "best"),
        ("--num_updates", "1500"),
        ("--episodeLength", "2600"),
        ("--warmup_iters", "50"),
        ("--ratio_range", "[0.2, 1]"),
        ("--cmdp_method", "np3o"),
        ("--lr", "2e-4"),
        ("--gamma", "1.0"),
        ("--value_bootstrap", "false"),
        ("--c_gamma", "[1, 0.99]"),
        ("--c_scale", "[0, 1]"),
        ("--successRewardScale", "1000"),
        ("--epstimeRewardScale", "[100, 100]"),
    )


def test_real_robot_examples_load_the_parent_calibration_bank():
    for relative in ("README.md", "docs/real_robot.md"):
        source = (ROOT / relative).read_text()
        real_robot_example = source.split("--real_robot", 1)[1].split("~~~", 1)[0]
        assert "--par_configs_eval true" in real_robot_example


def test_legacy_positional_stage_arguments_remain_supported(tmp_path):
    arguments = _captured_python_arguments(
        tmp_path,
        "train/temporal_student.sh",
        "FrankaCabinet", "teacher_run", "--seed", "9",
    )
    task = arguments.index("--task_name")
    checkpoint = arguments.index("--checkpoint")
    assert arguments[task:task + 2] == ["--task_name", "FrankaCabinet"]
    assert arguments[checkpoint:checkpoint + 2] == ["--checkpoint", "teacher_run"]
    assert arguments[-2:] == ["--seed", "9"]
