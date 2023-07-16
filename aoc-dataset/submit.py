#!/usr/bin/env python3
"""Submits a solution to be checked."""

import importlib.util

import argparse
from pathlib import Path


def import_from_path(path: str):
    """Imports a module from a path."""
    module_spec = importlib.util.spec_from_file_location("module", path)
    assert module_spec is not None, f"Could not find module at {path}"
    assert module_spec.loader is not None, f"Could not load module at {path}"
    module = importlib.util.module_from_spec(module_spec)
    assert module is not None, f"Could not create module at {path}"
    module_spec.loader.exec_module(module)

    return module


def submit(file: str):
    """Runs a submission against the test cases."""
    path = Path(file).absolute()
    module = import_from_path(str(path.with_name("test.py")))
    submission = import_from_path(str(path))

    results = []
    for test, output in module.TESTS:
        try:
            result = submission.solve(test)
            assert result == output, f"WA: Expected {output} got {result}"
        except Exception as e:
            results.append(e)

    return results


def main():
    """Runs the submission."""
    parser = argparse.ArgumentParser(description="AoC Dataset")
    parser.add_argument(
        "-f", "--file", type=str, required=True, help="The submission to run"
    )
    args = parser.parse_args()

    results = submit(args.file)

    if not results:
        print("OK")
    else:
        print(results)


if __name__ == "__main__":
    main()
