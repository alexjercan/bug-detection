import os
import logging
from models import make_pipeline
from metric import make_metric
from tqdm.auto import tqdm
from evaluate import load
from dataset import make_dataset
from util import generate_html_output, compute_bug_type
from args import parse_args, Options


dir_path = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(os.path.abspath(os.path.join(dir_path, os.pardir)), "input")
LOG_PATH = os.path.join(
    os.path.abspath(os.path.join(dir_path, os.pardir)), "repair.log"
)
ROOT_PATH = os.path.join(INPUT_PATH, "Project_CodeNet")
DERIVED_PATH = os.path.join(ROOT_PATH, "derived")
GENERATED_PATH = os.path.join(INPUT_PATH, "repair")


def main(options: Options):
    logging.info("Options: %s", options)

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
    evaluation_data = dataset.select(range(1)) if debug else dataset

    n = len(evaluation_data)
    indices = list(range(n))
    evaluation_data = evaluation_data.add_column("index", indices)

    # Make the metric
    code_eval = make_metric(dataset_path)

    # Generate the predictions of the code model on the eval data
    predictions = []
    for example in tqdm(evaluation_data, desc="Generating predictions"):
        try:
            prediction = pipe(example)
        except Exception as e:
            logging.error("Error: %s", e)
            prediction = [""]

        predictions.append(prediction)

    evaluation_data = evaluation_data.add_column("predicted", predictions)

    # Compute the exact match accuracy of the best prediction in the returned sequence
    exact_match = load("exact_match")
    result = exact_match.compute(
        predictions=[p[0] for p in evaluation_data["predicted"]], references=evaluation_data["pass"]
    )
    logging.info("Exact Match: %s", result)

    result, test_results = code_eval(evaluation_data, num_sequences, timeout)
    logging.info("Code Eval: %s", result)

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
    logging.info("Bug Type: %s", result)

    # Save the evaluation data
    data_name = dataset_path.split("/")[-1]
    model_name = checkpoint.split("/")[-1]
    out_path = os.path.join(GENERATED_PATH, f"evaluation_data_{data_name}_{model_name}.data")
    evaluation_data.save_to_disk(out_path)

    return evaluation_data


if __name__ == "__main__":
    args = parse_args()

    handler = logging.StreamHandler()
    logging.basicConfig(
        handlers=[handler, logging.FileHandler(LOG_PATH)],
        level=args.loglevel,
    )

    main(args)
