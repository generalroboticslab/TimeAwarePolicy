import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests" / "release" / "smoke_tasks.py"
SPEC = importlib.util.spec_from_file_location("smoke_tasks", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_smoke_command_is_small_non_saving_end_to_end_update(tmp_path):
    command = smoke.training_command(ROOT, "FrankaCubeStack", 32, 4, tmp_path)

    assert command[0] == sys.executable
    assert command[1:4] == [
        "-m",
        "projects.TimeAwarePolicy.train",
        "--task_name",
    ]
    assert command[command.index("--num_updates") + 1] == "1"
    assert command[command.index("--update_epochs") + 1] == "1"
    assert command[command.index("--minibatch-size") + 1] == "128"
    assert command[command.index("--saving") + 1] == "false"
    assert command[command.index("--wandb") + 1] == "false"
    assert "--nographics" in command


def test_smoke_default_covers_every_public_task():
    assert smoke.TASKS == (
        "FrankaCubeStack",
        "FrankaGmPour",
        "FrankaCabinet",
    )


def test_runtime_environment_uses_selected_python_prefix(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")

    environment = smoke.runtime_environment(ROOT, 7)
    python_bin = Path(sys.executable).resolve().parent

    assert environment["CUDA_VISIBLE_DEVICES"] == "7"
    assert environment["PATH"].split(":", 1)[0] == str(python_bin)
    assert environment["LD_LIBRARY_PATH"].split(":", 1)[0] == str(
        python_bin.parent / "lib"
    )
    assert environment["PYTHONPATH"].split(":") == [
        str(ROOT / "isaacgym" / "python"),
        str(ROOT / "envs"),
        str(ROOT),
    ]
