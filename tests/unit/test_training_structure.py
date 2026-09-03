import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_training_support_methods_are_extracted_once():
    method_names = {
        "update_curriculum",
        "log_episode_metrics",
        "_log_training_metrics",
        "save_checkpoints",
        "save_quality_candidate",
        "print_status",
        "print_student_status",
    }
    paths = (
        REPOSITORY_ROOT / "core" / "training" / "trainer.py",
        REPOSITORY_ROOT / "core" / "training" / "curriculum.py",
        REPOSITORY_ROOT / "core" / "training" / "logging.py",
        REPOSITORY_ROOT / "core" / "training" / "checkpointing.py",
        REPOSITORY_ROOT / "core" / "training" / "rollout.py",
    )
    definitions = []
    for path in paths:
        tree = ast.parse(path.read_text())
        definitions.extend(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in method_names
        )
    assert sorted(definitions) == sorted(method_names)


def test_rollout_methods_live_in_rollout_module_once():
    method_names = {
        "collect_rollout",
        "_update_episode_stats",
        "compute_advantages",
    }
    paths = (
        REPOSITORY_ROOT / "core" / "training" / "trainer.py",
        REPOSITORY_ROOT / "core" / "training" / "rollout.py",
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


def test_training_does_not_embed_a_wandb_account():
    training_source = (REPOSITORY_ROOT / "core" / "training" / "trainer.py").read_text()
    arguments_source = (
        REPOSITORY_ROOT
        / "projects"
        / "TimeAwarePolicy"
        / "arguments"
        / "training.py"
    ).read_text()

    assert "jiayinsen" not in training_source
    assert "--wandb_entity" in arguments_source
    assert "--wandb_project" in arguments_source


def test_quality_candidate_is_saved_before_curriculum_expands():
    training_source = (REPOSITORY_ROOT / "core" / "training" / "trainer.py").read_text()
    candidate_call = training_source.index("self.save_quality_candidate()")
    curriculum_call = training_source.index("self.update_curriculum()", candidate_call)

    assert candidate_call < curriculum_call
