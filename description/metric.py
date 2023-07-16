from args import DATASET_AOC, DATASET_BUGNET
from evaluate import load
from typing import Dict, List, Tuple

from util import compute_bug_type


def compute_eval_metric_bugnet(examples: Dict[str, List], num_sequences: int) -> Tuple:
    # TODO: Implement this
    _ = num_sequences

    # Compute the bug type (input, output, algorithm) using the pass code
    pass_bug_type = compute_bug_type(examples, "pass")["pass_bug_type"]
    predicted_bug_type = compute_bug_type(examples, "predicted")["predicted_bug_type"]

    # Compute the exact match accuracy of the bug type
    exact_match = load("exact_match")
    result = exact_match.compute(
        predictions=predicted_bug_type, references=pass_bug_type
    )

    test_results = {
        index: (index, {"passed": pass_ == predicted})
        for index, pass_, predicted in zip(
            examples["index"], pass_bug_type, predicted_bug_type
        )
    }

    return result, test_results


def compute_eval_metric_aoc(examples: Dict[str, List], num_sequences: int) -> Tuple:
    # TODO: Implement this
    _ = num_sequences

    # Compute the bug type (input, output, algorithm) using the pass code
    pass_bug_type = compute_bug_type(examples, "pass")["pass_bug_type"]
    predicted_bug_type = compute_bug_type(examples, "predicted")["predicted_bug_type"]

    # Compute the exact match accuracy of the bug type
    exact_match = load("exact_match")
    result = exact_match.compute(
        predictions=predicted_bug_type, references=pass_bug_type
    )

    test_results = {
        index: (index, {"passed": pass_ == predicted})
        for index, pass_, predicted in zip(
            examples["index"], pass_bug_type, predicted_bug_type
        )
    }

    return result, test_results


class Metric:
    def __init__(self):
        pass

    def __call__(self, examples: Dict[str, List], num_sequences: int) -> Tuple:
        raise NotImplementedError("This method must be implemented by a subclass.")


class BugNetMetric(Metric):
    def __call__(self, examples: Dict[str, List], num_sequences: int) -> Tuple:
        return compute_eval_metric_bugnet(examples, num_sequences)


class AoCMetric(Metric):
    def __call__(self, examples: Dict[str, List], num_sequences: int) -> Tuple:
        return compute_eval_metric_aoc(examples, num_sequences)


def make_metric(dataset_path: str) -> Metric:
    if dataset_path == DATASET_BUGNET:
        return BugNetMetric()

    if dataset_path == DATASET_AOC:
        return AoCMetric()

    raise ValueError(f"Unknown dataset: {dataset_path}")
