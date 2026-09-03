"""Small JSON and CSV helpers shared by training and evaluation."""

from collections import OrderedDict
import csv
import json
import os
import shutil


def read_json(json_path):
    with open(json_path, "r") as stream:
        return json.load(stream)


def save_json(data, json_path):
    with open(json_path, "w") as stream:
        json.dump(data, stream, indent=4)


def write_csv_line(result_file_path, result):
    """Append a dictionary row, creating a header for a new CSV file."""
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, "a", newline="") as stream:
        writer = csv.DictWriter(stream, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def check_file_exist(file_path):
    """Interactively remove an existing evaluation output or abort."""
    if not os.path.exists(file_path):
        return
    response = input(
        f"Find existing dir/file {file_path}! Whether remove or not (y/n):"
    )
    if response.lower() != "y":
        raise RuntimeError("Evaluation aborted because the output already exists.")
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)
    else:
        os.remove(file_path)
