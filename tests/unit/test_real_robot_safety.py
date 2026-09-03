import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _is_name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _is_not_move_guard(test):
    return any(
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Attribute)
        and node.operand.attr == "not_move"
        and isinstance(node.operand.value, ast.Attribute)
        and node.operand.value.attr == "args"
        and _is_name(node.operand.value.value, "self")
        for node in ast.walk(test)
    )


def test_joint_fk_clamps_joint_delta_with_symmetric_velocity_limits():
    source = (
        REPOSITORY_ROOT
        / "envs"
        / "isaacgymenvs"
        / "tasks"
        / "base"
        / "vec_task.py"
    )
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_joint_fk"
    )
    clamps = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and _is_name(node.func, "tensor_clamp")
        and len(node.args) >= 3
        and _is_name(node.args[0], "dq")
    ]

    assert any(
        isinstance(call.args[1], ast.UnaryOp)
        and isinstance(call.args[1].op, ast.USub)
        and _is_name(call.args[1].operand, "dq_max_abs")
        and _is_name(call.args[2], "dq_max_abs")
        for call in clamps
    )


def test_not_move_guards_every_real_robot_command_publication():
    standalone = REPOSITORY_ROOT / "real_robot" / "evaluation.py"
    source = (
        standalone
        if standalone.is_file()
        else REPOSITORY_ROOT / "core" / "evaluation" / "evaluator.py"
    )
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "evaluate_real_robot"
    )
    parents = {}
    for parent in ast.walk(function):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    publications = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "send_command"
    ]

    assert publications
    for publication in publications:
        ancestor = parents.get(publication)
        guarded = False
        while ancestor is not None and ancestor is not function:
            if isinstance(ancestor, ast.If) and _is_not_move_guard(ancestor.test):
                guarded = True
                break
            ancestor = parents.get(ancestor)
        assert guarded
