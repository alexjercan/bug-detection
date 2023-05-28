import argparse
from dataclasses import dataclass
import os

from tqdm.auto import tqdm
from typing import Dict, Tuple, List
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
import json
from typing import Optional
import subprocess
import resource
import uuid

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

MAX_VIRTUAL_MEMORY = 100 * 1024 * 1024  # 100 MB
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Options:
    """Options for the script."""

    debug: bool = False
    dataset: str = "alexjercan/bugnet"
    subset: str = "Python"
    checkpoint: str = "Salesforce/codegen2-1B"
    timeout: float = 2
    num_sequences: int = 2
    split: str = "test"


def color_source(source_code: str, i1: int, i2: int, color: str = "red"):
    lines = source_code.splitlines(keepends=True)

    text = ""
    for i, line_str in enumerate(lines):
        for char in line_str:
            norm_color = 'black'
            if char == ' ':
                char = "•"
                norm_color = 'lightgrey'
            if char == '\n':
                char = "↵\n"
                norm_color = 'lightgrey'
            text += f'<span style="color:{color if i1 <= i and i <= i2 else norm_color};">{char}</span>'

    return "<pre>" + text + "</pre>"


def generate_html_output(
    examples: Dict[str, List], test_results: Dict[int, List]
) -> Dict[str, List]:
    # Display the source code. First show the buggy submission with red lines for the bug
    # Then show the passing submission with green lines. Then show the predictions
    # with red lines if the change did not work, and with green lines if the change
    # made the tests pass
    results = []
    passed = []
    for i, (fail, pass_, i1, i2, j1, j2, predictions) in enumerate(zip(examples["fail"], examples["pass"], examples["i1"], examples["i2"], examples["j1"], examples["j2"], examples["predicted"])):
        fail_html = color_source(fail, i1, i2, color="red")
        pass_html = color_source(pass_, j1, j2, color="green")

        html = ""
        html += f"<h2>Example {i}</h2>"

        html += "<h6>Original Source Code</h6>"
        html += fail_html

        html += "<h6>Changed Source Code</h6>"
        html += pass_html

        any_correct = False
        for j, (pred, (_, test)) in enumerate(zip(predictions, test_results[i])):
            color = "green" if test["passed"] else "red"
            diff_len = len(pred.splitlines()) - len(fail.splitlines())
            pred_html = color_source(pred, i1, i2 + diff_len, color=color)

            html += f"<h6>Predicted Source Code {j}</h6>"
            html += pred_html

            any_correct = any_correct or test["passed"]

        results.append(html)
        passed.append(any_correct)

    return {"html": results, "any_correct": passed}


def execute_source(
    source_code: str, language: str, input: str, timeout: Optional[float] = None
) -> Optional[str]:
    path = f"/tmp/{uuid.uuid4}.xxx"
    with open(path, "w") as f:
        f.write(source_code)

    if language == "C++":
        out = f"/tmp/{uuid.uuid4()}.out"

        result = subprocess.run(["g++", path, "-o", out], capture_output=True)

        if result.returncode != 0:
            return None

        try:
            result = subprocess.run(
                [out],
                input=input,
                timeout=timeout,
                capture_output=True,
                encoding="utf-8",
                errors="ignore",
                preexec_fn=lambda: resource.setrlimit(
                    resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY, resource.RLIM_INFINITY)
                ),
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return None
    if language == "Python":
        try:
            result = subprocess.run(
                ["python3", path],
                input=input,
                timeout=timeout,
                capture_output=True,
                encoding="utf-8",
                errors="ignore",
                preexec_fn=lambda: resource.setrlimit(
                    resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY, resource.RLIM_INFINITY)
                ),
            )

            return result.stdout
        except subprocess.TimeoutExpired:
            return None

    raise NotImplementedError(f"{language} not implemented yet")


class CodeGenEvalPipeline:
    def __init__(self, tokenizer, model, num_sequences=2):
        self.tokenizer = tokenizer
        self.model = model
        self.num_sequences = num_sequences

    def __call_one__(self, text):
        tokenized_input = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        tokenized_input = tokenized_input.to(DEVICE)

        with torch.no_grad():
            output_ids = self.model.generate(
                **tokenized_input,
                max_new_tokens=512,
                num_beams=self.num_sequences,
                num_return_sequences=self.num_sequences,
                early_stopping=True,
            )

        predictions = []
        for i in range(output_ids.shape[0]):
            mask_prediction = self.tokenizer.decode(output_ids[i])[len(text):].split("<eom>")[0]
            predictions.append(
                text.replace("<mask_1>", mask_prediction).split("<|endoftext|>")[0]
            )

        return predictions

    def __call__(self, texts, **kwargs):
        return [self.__call_one__(text) for text in texts]


def format_example_batched(example: Dict[str, List]) -> Dict[str, List]:
    original_srcs = [src.splitlines(keepends=False) for src in example["fail"]]
    changed_srcs = [src.splitlines(keepends=False) for src in example["pass"]]
    i1s = example["i1"]
    i2s = example["i2"]
    j1s = example["j1"]
    j2s = example["j2"]

    prefixes = [
        "\n".join(original_src[:i1]) for original_src, i1 in zip(original_srcs, i1s)
    ]
    suffixes = [
        "\n".join(original_src[i2:]) for original_src, i2 in zip(original_srcs, i2s)
    ]
    changes = [
        "\n".join(changed_src[j1:j2])
        for changed_src, j1, j2 in zip(changed_srcs, j1s, j2s)
    ]

    return {
        "text": [
            prefix + "\n<mask_1>\n" + suffix + "<|endoftext|>" + "<sep>" + "<mask_1>"
            for prefix, suffix in zip(prefixes, suffixes)
        ],
        "label": changes,
    }


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


def compute_eval_metric_bugnet(evaluation_data, num_sequences: int, timeout: float):
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
    k = [1, num_sequences]
    result, test_results = code_eval.compute(
        predictions=evaluation_data["assertion"], references=["" for _ in evaluation_data["assertion"]], k=k
    )
    print(f"Code Eval: {result}")

    # Generate the html output for the evaluation data
    evaluation_data = evaluation_data.map(
        generate_html_output, batched=True, num_proc=4, fn_kwargs={"test_results": test_results}
    )

    return evaluation_data


def compute_eval_metric_aoc(evaluation_data, num_sequences: int, timeout: float):
    # Compute the code eval (pass@k) for the predictions of the model
    code_eval = load("code_eval")
    k = [1, num_sequences]
    result, test_results = code_eval.compute(
        predictions=evaluation_data["predicted"], references=evaluation_data["test"], k=k
    )
    print(f"Code Eval: {result}")

    # Generate the html output for the evaluation data
    evaluation_data = evaluation_data.map(
        generate_html_output, batched=True, num_proc=4, fn_kwargs={"test_results": test_results}
    )

    return evaluation_data


def compute_bug_type_pass(example: Dict[str, List]) -> Dict[str, List]:
    results = []
    for j1, j2, pass_, language in zip(example["j1"], example["j2"], example["pass"], example["language"]):
        line = "\n".join(pass_.splitlines()[j1:j2])

        if language == "Python":
            results.append(
                "input"
                if "input" in line
                else "output"
                if "print" in line
                else "algorithm"
            )
        elif language == "C++":
            results.append(
                "input"
                if ("cin" in line or "scanf" in line)
                else "output"
                if ("cout" in line or "printf" in line)
                else "algorithm"
            )
        else:
            raise NotImplementedError(f"{language} not implemented yet")

    return {"pass_bug_type": results}


def compute_bug_type_predicted(example: Dict[str, List]) -> Dict[str, List]:
    results = []
    for i1, i2, fail, predicted, language in zip(example["i1"], example["i2"], example["fail"], example["predicted"], example["language"]):
        diff_len = len(predicted[0].splitlines()) - len(fail.splitlines())
        line = "\n".join(predicted[0].splitlines()[i1:i2 + diff_len])

        if language == "Python":
            results.append(
                "input"
                if "input" in line
                else "output"
                if "print" in line
                else "algorithm"
            )
        elif language == "C++":
            results.append(
                "input"
                if ("cin" in line or "scanf" in line)
                else "output"
                if ("cout" in line or "printf" in line)
                else "algorithm"
            )
        else:
            raise NotImplementedError(f"{language} not implemented yet")

    return {"predicted_bug_type": results}


def main(options: Options):
    # Load the options
    debug = options.debug
    checkpoint = options.checkpoint
    dataset_path = options.dataset
    subset = options.subset
    timeout = options.timeout
    num_sequences = options.num_sequences
    split = options.split

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, revision="main"
    )
    model = model.to(DEVICE)

    # Create the pipeline that will output the infilled code. The input to this
    # pipeline will be a list of examples of source code where each example should
    # have the following format [prefix]<mask_1>[suffix]<|endoftext|><sep><mask_1>
    # This will prompt the model to generate the text for the <mask_1> token.
    # The pipeline will return the replaced source code.
    model_name = checkpoint.split("/")[-1]
    if model_name == "codegen2-1B":
        pipe = CodeGenEvalPipeline(tokenizer, model, num_sequences=num_sequences)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Load the dataset
    dataset = load_dataset(dataset_path, subset)
    evaluation_data = dataset[split].select(range(1)) if debug else dataset[split]

    data_name = dataset_path.split("/")[-1]
    if data_name == "AoC":
        evaluation_data = evaluation_data.add_column("language", ["Python" for _ in evaluation_data["pass"]])

    # Format the examples to be used by the code model
    evaluation_data = evaluation_data.map(format_example_batched, batched=True, num_proc=4)

    # Generate the predictions of the code model on the eval data
    predictions = []
    for example in tqdm(evaluation_data, desc="Generating predictions"):
        prediction = pipe([example["text"]])
        predictions.append(prediction[0])

    evaluation_data = evaluation_data.add_column("predicted", predictions)

    # Compute the exact match accuracy of the best prediction in the returned sequence
    exact_match = load("exact_match")
    result = exact_match.compute(
        predictions=[p[0] for p in evaluation_data["predicted"]], references=evaluation_data["pass"]
    )
    print(f"Exact Match: {result}")

    if data_name == "bugnet":
        # Compute the eval metric for the bugnet dataset
        evaluation_data = compute_eval_metric_bugnet(evaluation_data, num_sequences, timeout)
    elif data_name == "AoC":
        # Compute the eval metric for the AoC dataset
        evaluation_data = compute_eval_metric_aoc(evaluation_data, num_sequences, timeout)
    else:
        print(f"Cannot compute eval metric for dataset: {data_name}. Skipping...")

    # Compute the bug type (input, output, algorithm) using the pass code
    evaluation_data = evaluation_data.map(
        compute_bug_type_pass, batched=True, num_proc=4
    )

    # Compute the bug type (input, output, algorithm) using the predicted code
    evaluation_data = evaluation_data.map(
        compute_bug_type_predicted, batched=True, num_proc=4
    )

    # Compute the exact match accuracy of the bug type
    exact_match = load("exact_match")
    result = exact_match.compute(
        predictions=[p[0] for p in evaluation_data["predicted_bug_type"]], references=evaluation_data["pass_bug_type"]
    )
    print(f"Bug Type: {result}")

    # Save the evaluation data
    evaluation_data.save_to_disk(f"evaluation_data_{data_name}_{model_name}")

    return evaluation_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate a code generation model")

    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--dataset", type=str, default="alexjercan/bugnet", help="Dataset to use")
    parser.add_argument("--subset", type=str, default="Python", help="Subset to use")
    parser.add_argument("--checkpoint", type=str, default="Salesforce/codegen2-1B", help="Checkpoint to use")
    parser.add_argument("--timeout", type=float, default=2, help="Timeout for the execution of the generated code")
    parser.add_argument("--num_sequences", type=int, default=2, help="Number of sequences to generate")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate on")

    args = parser.parse_args()

    main(
        Options(
            debug=args.debug,
            dataset=args.dataset,
            subset=args.subset,
            checkpoint=args.checkpoint,
            timeout=args.timeout,
            num_sequences=args.num_sequences,
            split=args.split,
        )
    )
