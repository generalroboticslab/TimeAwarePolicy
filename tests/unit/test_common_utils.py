import numpy as np
import torch

from core.common.io import read_json, save_json
from core.common.tensor import to_numpy
from core.common.time import convert_time, format_timestamp


def test_runtime_formatting_and_argument_helpers():
    assert convert_time(3661.9) == "1:1:1"
    assert format_timestamp(62.3456) == "01:02.345"
    assert format_timestamp(-0.25) == "-00:00.250"


def test_runtime_json_and_numpy_helpers(tmp_path):
    path = tmp_path / "value.json"
    save_json({"answer": 42}, path)
    assert read_json(path) == {"answer": 42}
    assert np.array_equal(to_numpy(torch.tensor(3.0)), np.array([3.0]))
    assert np.array_equal(to_numpy([1, 2]), np.array([1, 2]))
