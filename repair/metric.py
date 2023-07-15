import math
from typing import Dict, List, Optional, Tuple
import subprocess
import resource
import uuid
from evaluate import load
import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
MAX_VIRTUAL_MEMORY = 100 * 1024 * 1024  # 100 MB


def execute_source(
    source_code: str, language: str, input_: str, timeout: Optional[float] = None
) -> Optional[str]:
    path = f"/tmp/{uuid.uuid4}.xxx"
    with open(path, "w", encoding="utf-8") as f:
        f.write(source_code)

    if language == "C++":
        out = f"/tmp/{uuid.uuid4()}.out"

        result = subprocess.run(["g++", path, "-o", out], capture_output=True, check=True)

        if result.returncode != 0:
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
                    resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY, resource.RLIM_INFINITY)
                ),
                check=True,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
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
                    resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY, resource.RLIM_INFINITY)
                ),
                check=True,
            )

            return result.stdout
        except subprocess.TimeoutExpired:
            return None

    raise NotImplementedError(f"{language} not implemented yet")


def generate_execution_results(
    example: Dict[str, List], timeout: Optional[float] = None
) -> Dict[str, List]:
    results = []
    for prediction, input_, language in zip(example["predicted"], example["input"], example["language"]):
        prediction_results = []
        for source in prediction:
            execute_output = execute_source(source, language, input_, timeout=timeout)
            prediction_results.append(execute_output)

        results.append(prediction_results)

    return {"execution": results}


def generate_powers_of_two(k: int) -> List[int]:
    max_exponent = math.floor(math.log2(k))

    return [2 ** i for i in range(max_exponent + 1)]


def generate_assertion_statements(example: Dict[str, List]) -> Dict[str, List]:
    results = []
    for execution, output in zip(example["execution"], example["output"]):
        assertion_results = []
        for execution_output in execution:
            assertion_results.append(
                f"assert {repr(execution_output)} == {repr(output)}"
            )

        results.append(assertion_results)

    return {"assertion": results}


def compute_eval_metric_bugnet(evaluation_data, num_sequences: int, timeout: float) -> Tuple:
    # Generate the execution output of the predicted source code
    evaluation_data = evaluation_data.map(
        generate_execution_results, batched=True, num_proc=4, fn_kwargs={"timeout": timeout}
    )

    # Generate the assertion statements for the execution output
    evaluation_data = evaluation_data.map(
        generate_assertion_statements, batched=True, num_proc=4
    )

    # Compute the code eval (pass@k) for the predictions of the model
    code_eval = load("code_eval")
    k = generate_powers_of_two(num_sequences)
    return code_eval.compute(
        predictions=evaluation_data["assertion"], references=["" for _ in evaluation_data["assertion"]], k=k
    )


def compute_eval_metric_aoc(evaluation_data, num_sequences: int, timeout: float) -> Tuple:
    # Compute the code eval (pass@k) for the predictions of the model
    code_eval = load("code_eval")
    k = generate_powers_of_two(num_sequences)
    return code_eval.compute(
        predictions=evaluation_data["predicted"], references=evaluation_data["test"], k=k
    )


class Metric:
    def __init__(self):
        pass

    def __call__(self, examples: Dict[str, List], num_sequences: int, timeout: float) -> Tuple:
        raise NotImplementedError("This method must be implemented by a subclass.")


class BugNetMetric(Metric):
    def __call__(self, examples: Dict[str, List], num_sequences: int, timeout: float) -> Tuple:
        return compute_eval_metric_bugnet(examples, num_sequences, timeout)


class AoCMetric(Metric):
    def __call__(self, examples: Dict[str, List], num_sequences: int, timeout: float) -> Tuple:
        return compute_eval_metric_aoc(examples, num_sequences, timeout)
