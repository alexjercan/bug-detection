import os

import math
import resource
import subprocess
import uuid
from args import DATASET_AOC, DATASET_BUGNET
from evaluate import load
from typing import Dict, List, Optional, Tuple

from util import compute_bug_type

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
MAX_VIRTUAL_MEMORY = 100 * 1024 * 1024  # 100 MB


def execute_source(
    source_code: str,
    language: str,
    input_: str,
    timeout: Optional[float] = None,
) -> Optional[str]:
    path = f"/tmp/{uuid.uuid4}.xxx"
    with open(path, "w", encoding="utf-8") as f:
        f.write(source_code)

    if language == "C++":
        out = f"/tmp/{uuid.uuid4()}.out"

        try:
            subprocess.run(["g++", path, "-o", out], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            return None

        try:
            result = subprocess.run(
                [out],
                input=input_,
                timeout=timeout,
                capture_output=True,
                encoding="utf-8",
                errors="ignore",
                preexec_fn=lambda: resource.setrlimit(
                    resource.RLIMIT_AS,
                    (MAX_VIRTUAL_MEMORY, resource.RLIM_INFINITY),
                ),
                check=True,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return None
        except subprocess.CalledProcessError:
            return None
    if language == "Python":
        try:
            result = subprocess.run(
                ["python3", path],
                input=input_,
                timeout=timeout,
                capture_output=True,
                encoding="utf-8",
                errors="ignore",
                preexec_fn=lambda: resource.setrlimit(
                    resource.RLIMIT_AS,
                    (MAX_VIRTUAL_MEMORY, resource.RLIM_INFINITY),
                ),
                check=True,
            )

            return result.stdout
        except subprocess.TimeoutExpired:
            return None
        except subprocess.CalledProcessError:
            return None

    raise NotImplementedError(f"{language} not implemented yet")


def generate_execution_results(
    example: Dict[str, List], timeout: Optional[float] = None
) -> Dict[str, List]:
    results = []
    for prediction, input_, language in zip(
        example["predicted"], example["input"], example["language"]
    ):
        prediction_results = []
        for source in prediction:
            execute_output = execute_source(source, language, input_, timeout=timeout)
            prediction_results.append(execute_output)

        results.append(prediction_results)

    return {"execution": results}


def generate_powers_of_two(k: int) -> List[int]:
    max_exponent = math.floor(math.log2(k))

    return [2**i for i in range(max_exponent + 1)]


def generate_assertion_statements(
    example: Dict[str, List], executions: List
) -> Dict[str, List]:
    results = []
    for execution, output in zip(executions, example["output"]):
        assertion_results = []
        for execution_output in execution:
            assertion_results.append(
                f"assert {repr(execution_output)} == {repr(output)}"
            )

        results.append(assertion_results)

    return {"assertion": results}


def compute_eval_metric_bugnet(
    examples: Dict[str, List], num_sequences: int, timeout: float
) -> Tuple:
    # Generate the execution output of the predicted source code
    execution = generate_execution_results(examples, timeout=timeout)["execution"]

    # Generate the assertion statements for the execution output
    assertion = generate_assertion_statements(examples, execution)["assertion"]

    # Compute the code eval (pass@k) for the predictions of the model
    code_eval = load("code_eval")
    k = generate_powers_of_two(num_sequences)
    result1_tup = code_eval.compute(
        predictions=assertion, references=["" for _ in assertion], k=k
    )

    assert isinstance(
        result1_tup, tuple
    ), f"The result of the evaluation must be a tuple, but got {type(result1_tup)}"

    result1, test_results = result1_tup

    # Compute the exact match accuracy of the best prediction in the returned sequence
    exact_match = load("exact_match")
    result2 = exact_match.compute(
        predictions=[p[0] for p in examples["predicted"]],
        references=examples["pass"],
    )

    assert result2 is not None, "The exact match accuracy must not be None"
    result2 = {"exact_match": result2["exact_match"]}

    # Compute the exact match accuracy of the bug type
    pass_bug_type = compute_bug_type(examples, "pass")["pass_bug_type"]
    predicted_bug_type = compute_bug_type(examples, "predicted")["predicted_bug_type"]

    exact_match = load("exact_match")
    result3 = exact_match.compute(
        predictions=predicted_bug_type, references=pass_bug_type
    )

    assert result3 is not None, "The exact match accuracy must not be None"
    result3 = {"bug_type": result3["exact_match"]}

    result = {**result1, **result2, **result3}

    return result, test_results


def compute_eval_metric_aoc(examples: Dict[str, List], num_sequences: int) -> Tuple:
    # Compute the code eval (pass@k) for the predictions of the model
    code_eval = load("code_eval")
    k = generate_powers_of_two(num_sequences)
    result1_tup = code_eval.compute(
        predictions=examples["predicted"],
        references=examples["test"],
        k=k,
    )

    assert isinstance(
        result1_tup, tuple
    ), f"The result of the evaluation must be a tuple, but got {type(result1_tup)}"

    result1, test_results = result1_tup

    # Compute the exact match accuracy of the best prediction in the returned sequence
    exact_match = load("exact_match")
    result2 = exact_match.compute(
        predictions=[p[0] for p in examples["predicted"]],
        references=examples["pass"],
    )

    assert result2 is not None, "The exact match accuracy must not be None"
    result2 = {"exact_match": result2["exact_match"]}

    # Compute the exact match accuracy of the bug type
    pass_bug_type = compute_bug_type(examples, "pass")["pass_bug_type"]
    predicted_bug_type = compute_bug_type(examples, "predicted")["predicted_bug_type"]

    exact_match = load("exact_match")
    result3 = exact_match.compute(
        predictions=predicted_bug_type, references=pass_bug_type
    )

    assert result3 is not None, "The exact match accuracy must not be None"
    result3 = {"bug_type": result3["exact_match"]}

    result = {**result1, **result2, **result3}

    return result, test_results


class Metric:
    def __init__(self):
        pass

    def __call__(
        self, examples: Dict[str, List], num_sequences: int, timeout: float
    ) -> Tuple:
        raise NotImplementedError("This method must be implemented by a subclass.")


class BugNetMetric(Metric):
    def __call__(
        self, examples: Dict[str, List], num_sequences: int, timeout: float
    ) -> Tuple:
        return compute_eval_metric_bugnet(examples, num_sequences, timeout)


class AoCMetric(Metric):
    def __call__(
        self, examples: Dict[str, List], num_sequences: int, timeout: float
    ) -> Tuple:
        return compute_eval_metric_aoc(examples, num_sequences)


def make_metric(dataset_path: str) -> Metric:
    if dataset_path == DATASET_BUGNET:
        return BugNetMetric()

    if dataset_path == DATASET_AOC:
        return AoCMetric()

    raise ValueError(f"Unknown dataset: {dataset_path}")
