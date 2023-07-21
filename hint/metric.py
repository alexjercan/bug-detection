# TODO: this is a hack

from args import DATASET_AOC, DATASET_BUGNET
from evaluate import load
from typing import Dict, List, Tuple

from util import compute_bug_type


def compute_eval_metric_bugnet(examples: Dict[str, List], num_sequences: int) -> Tuple:
    # compute the bleu score
    bleu = load("bleu")
    references = [
        ["\n".join(e.splitlines()[j1:j2]) + "\n" + r]
        for r, e, j1, j2 in zip(
            examples["error"], examples["pass"], examples["j1"], examples["j2"]
        )
        for _ in range(num_sequences)
    ]
    predictions = [e for es in examples["predicted"] for e in es]

    result1 = bleu.compute(predictions=predictions, references=references)
    assert result1 is not None, "The bleu score must not be None"

    # Compute the bug type (input, output, algorithm) using the pass code
    pass_bug_type = compute_bug_type(examples, "pass")["pass_bug_type"]
    predicted_bug_type = compute_bug_type(examples, "predicted")["predicted_bug_type"]

    pass_bug_type_long = [e for e in pass_bug_type for _ in range(num_sequences)]
    predicted_bug_type_flat = [e for es in predicted_bug_type for e in es]

    # Compute the exact match accuracy of the bug type
    exact_match = load("exact_match")
    result2 = exact_match.compute(
        predictions=predicted_bug_type_flat, references=pass_bug_type_long
    )
    assert result2 is not None, "The exact match accuracy must not be None"

    test_results = {
        index: [(index, {"passed": pass_ == pred}) for pred in predicted]
        for index, pass_, predicted in zip(
            examples["index"], pass_bug_type, predicted_bug_type
        )
    }

    result = {**result1, **result2}

    return result, test_results


def compute_eval_metric_aoc(examples: Dict[str, List], num_sequences: int) -> Tuple:
    # compute the bleu score
    bleu = load("bleu")
    references = [
        ["\n".join(e.splitlines()[j1:j2])]
        for e, j1, j2 in zip(examples["pass"], examples["j1"], examples["j2"])
        for _ in range(num_sequences)
    ]
    predictions = [e for es in examples["predicted"] for e in es]

    result1 = bleu.compute(predictions=predictions, references=references)
    assert result1 is not None, "The bleu score must not be None"

    # Compute the bug type (input, output, algorithm) using the pass code
    pass_bug_type = compute_bug_type(examples, "pass")["pass_bug_type"]
    predicted_bug_type = compute_bug_type(examples, "predicted")["predicted_bug_type"]

    pass_bug_type_long = [e for e in pass_bug_type for _ in range(num_sequences)]
    predicted_bug_type_flat = [e for es in predicted_bug_type for e in es]

    # Compute the exact match accuracy of the bug type
    exact_match = load("exact_match")
    result2 = exact_match.compute(
        predictions=predicted_bug_type_flat, references=pass_bug_type_long
    )
    assert result2 is not None, "The exact match accuracy must not be None"

    test_results = {
        index: [(index, {"passed": pass_ == pred}) for pred in predicted]
        for index, pass_, predicted in zip(
            examples["index"], pass_bug_type, predicted_bug_type
        )
    }

    result = {**result1, **result2}

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
