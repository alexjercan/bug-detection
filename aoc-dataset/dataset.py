#!/usr/bin/env python3
"""Walks the directory structure and generates a dataset."""

import importlib.util
import json
import os

from difflib import SequenceMatcher
from glob import glob
from typing import Optional, Tuple


def import_from_path(path: str):
    """Imports a module from a path."""
    module_spec = importlib.util.spec_from_file_location("module", path)
    assert module_spec is not None, f"Could not find module at {path}"
    assert module_spec.loader is not None, f"Could not load module at {path}"
    module = importlib.util.module_from_spec(module_spec)
    assert module is not None, f"Could not create module at {path}"
    module_spec.loader.exec_module(module)

    return module


def lines_diff_checker(
    original_src: str, changed_src: str
) -> Optional[Tuple[str, int, int, int, int]]:
    """Checks the diff between two strings and returns the diff as a single chunk."""
    original_src_lines = original_src.splitlines()
    changed_src_lines = changed_src.splitlines()

    opcodes = SequenceMatcher(None, original_src_lines, changed_src_lines).get_opcodes()
    changes = list(filter(lambda opcode: opcode[0] != "equal", opcodes))
    if not changes:
        return None

    if len(changes) == 1:
        return changes[0]

    _, i1, _, j1, _ = changes[0]
    _, _, i2, _, j2 = changes[-1]

    return "replace", i1, i2, j1, j2


def walk_dataset(years):
    """Walks the dataset and generates a list of problems."""
    dataset = []

    for year in years:
        year = str(year)

        for day in sorted(os.listdir(year)):
            day_path = os.path.join(year, day)

            for part in ["part1", "part2"]:
                part_path = os.path.join(day_path, part)
                module = import_from_path(os.path.join(part_path, "test.py"))

                test_cases = []
                for test, output in module.TESTS:
                    test_cases.append(f"assert solve({test!r}) == {output!r}")
                test_cases = "\n".join(test_cases)

                pass_paths = sorted(glob(os.path.join(part_path, "pass*.py")))
                fail_paths = sorted(glob(os.path.join(part_path, "fail*.py")))
                for pass_path, fail_path in zip(pass_paths, fail_paths):
                    with open(pass_path, "r", encoding="utf-8") as file:
                        pass_content = file.read()
                    with open(fail_path, "r", encoding="utf-8") as file:
                        fail_content = file.read()

                    diff = lines_diff_checker(fail_content, pass_content)
                    if diff is None:
                        print("Failed to find diff for", pass_path, fail_path)
                        continue
                    change, i1, i2, j1, j2 = diff

                    dataset.append(
                        {
                            "year": year,
                            "day": day,
                            "part": part,
                            "pass": pass_content,
                            "fail": fail_content,
                            "test": test_cases,
                            "change": change,
                            "i1": i1,
                            "i2": i2,
                            "j1": j1,
                            "j2": j2,
                        }
                    )

    return dataset


if __name__ == "__main__":
    DATASET = walk_dataset(years=[2022])

    with open("dataset.jsonl", "w", encoding="utf-8") as FILE:
        for DATA in DATASET:
            FILE.write(json.dumps(DATA) + "\n")
