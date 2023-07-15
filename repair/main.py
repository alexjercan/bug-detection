from models import Pipeline, CodeGen2Pipeline, ChatGPTPipeline
from metric import Metric, BugNetMetric, AoCMetric
import argparse
from dataclasses import dataclass
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from evaluate import load
from util import generate_html_output, compute_bug_type

DATASET_BUGNET = "alexjercan/bugnet"
DATASET_AOC = "alexjercan/AoC"

MODEL_CHATGPT = "gpt-3.5-turbo"
MODEL_CODEGEN2 = "codegens"


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


def make_pipeline(checkpoint: str) -> Pipeline:
    if checkpoint == MODEL_CODEGEN2:
        return CodeGen2Pipeline

    if checkpoint == MODEL_CHATGPT:
        return ChatGPTPipeline

    raise ValueError(f"Unknown checkpoint: {checkpoint}")


def make_dataset(dataset_path: str, subset: str, split: str) -> Dataset:
    dataset = load_dataset(dataset_path, subset)
    evaluation_data = dataset[split]

    if dataset_path == DATASET_AOC:
        evaluation_data = evaluation_data.add_column("language", ["Python" for _ in evaluation_data["pass"]])

    return evaluation_data


def make_metric(dataset_path: str) -> Metric:
    if dataset_path == DATASET_BUGNET:
        return BugNetMetric()

    if dataset_path == DATASET_AOC:
        return AoCMetric()

    raise ValueError(f"Unknown dataset: {dataset_path}")


def main(options: Options):
    # Load the options
    debug = options.debug
    checkpoint = options.checkpoint
    dataset_path = options.dataset
    subset = options.subset
    timeout = options.timeout
    num_sequences = options.num_sequences
    split = options.split

    # Create the pipeline
    pipe = make_pipeline(checkpoint)

    # Load the dataset
    dataset = make_dataset(dataset_path, subset, split)
    evaluation_data = dataset.select(range(1)) if debug else dataset[split]

    # Make the metric
    code_eval = make_metric(dataset_path)

    # Generate the predictions of the code model on the eval data
    predictions = []
    for example in tqdm(evaluation_data, desc="Generating predictions"):
        prediction = pipe([example])
        predictions.append(prediction[0])

    evaluation_data = evaluation_data.add_column("predicted", predictions)

    # Compute the exact match accuracy of the best prediction in the returned sequence
    exact_match = load("exact_match")
    result = exact_match.compute(
        predictions=[p[0] for p in evaluation_data["predicted"]], references=evaluation_data["pass"]
    )
    print(f"Exact Match: {result}")

    result, test_results = code_eval(evaluation_data, num_sequences, timeout)
    print(f"Code Eval: {result}")

    # Generate the html output for the evaluation data
    evaluation_data = evaluation_data.map(
        generate_html_output, batched=True, num_proc=4, fn_kwargs={"test_results": test_results}
    )

    # Compute the bug type (input, output, algorithm) using the pass code
    evaluation_data = evaluation_data.map(
        compute_bug_type, batched=True, num_proc=4, fn_kwargs={"which": "pass"}
    )

    # Compute the bug type (input, output, algorithm) using the predicted code
    evaluation_data = evaluation_data.map(
        compute_bug_type, batched=True, num_proc=4, fn_kwargs={"which": "predicted"}
    )

    # Compute the exact match accuracy of the bug type
    exact_match = load("exact_match")
    result = exact_match.compute(
        predictions=[p[0] for p in evaluation_data["predicted_bug_type"]], references=evaluation_data["pass_bug_type"]
    )
    print(f"Bug Type: {result}")

    # Save the evaluation data
    data_name = dataset_path.split("/")[-1]
    model_name = checkpoint.split("/")[-1]
    evaluation_data.save_to_disk(f"evaluation_data_{data_name}_{model_name}.data")

    return evaluation_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate a code generation model")

    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--dataset", type=str, default=DATASET_BUGNET, choices=[DATASET_BUGNET, DATASET_AOC], help="Dataset to use")
    parser.add_argument("--subset", type=str, default="Python", help="Subset to use")
    parser.add_argument("--checkpoint", type=str, default=MODEL_CHATGPT, choices=[MODEL_CHATGPT, MODEL_CODEGEN2], help="Checkpoint to use")
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
