import ast
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_real_robot_sources_parse_and_controller_host_is_explicit():
    for path in (REPOSITORY_ROOT / "real_robot").glob("*.py"):
        ast.parse(path.read_text())

    socket_tree = ast.parse(
        (REPOSITORY_ROOT / "real_robot" / "SocketClient.py").read_text()
    )
    client = next(
        node
        for node in socket_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FrankaClient"
    )
    initializer = next(
        node
        for node in client.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    parameter_names = [argument.arg for argument in initializer.args.args]
    first_default = len(parameter_names) - len(initializer.args.defaults)
    assert parameter_names.index("controller_ip") < first_default


def test_controller_warns_about_pickle_transport():
    pytest.importorskip("zmq")
    from real_robot.SocketClient import FrankaClient

    with pytest.warns(RuntimeWarning, match="trusted, isolated network"):
        client = FrankaClient(controller_ip="127.0.0.1")
    client.stop()
