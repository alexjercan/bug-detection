import json
import logging
import os

from args import Options, parse_args
from data import make_dataset
from metric import make_metric
from tqdm.auto import tqdm

from models import make_pipeline
from util import generate_html_output

LOGGER = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(os.path.abspath(os.path.join(dir_path, os.pardir)), "input")
LOG_PATH = os.path.join(os.path.abspath(os.path.join(dir_path, os.pardir)), "hint.log")
ROOT_PATH = os.path.join(INPUT_PATH, "Project_CodeNet")
DERIVED_PATH = os.path.join(ROOT_PATH, "derived")
GENERATED_PATH = os.path.join(INPUT_PATH, "hint")


def main(options: Options):
    logging.info("Options: %s", options)

    # Load the options
    debug = options.debug
    checkpoint = options.checkpoint
    dataset_path = options.dataset
    subset = options.subset
    num_sequences = options.num_sequences
    split = options.split

    # Create the pipeline
    pipe = make_pipeline(checkpoint, num_sequences=num_sequences)

    # Load the dataset
    dataset = make_dataset(dataset_path, subset, split)
    evaluation_data = dataset.select(range(1)) if debug else dataset

    n = len(evaluation_data)
    indices = list(range(n))
    evaluation_data = evaluation_data.add_column("index", indices)  # type: ignore

    # Make the metric
    eval_metric = make_metric(dataset_path)

    # Generate the predictions of the code model on the eval data
    predictions = []
    for example in tqdm(evaluation_data, desc="Generating predictions"):
        try:
            prediction = pipe(example)
        except Exception as e:
            logging.error("Error: %s", e)
            prediction = ["" for _ in range(num_sequences)]

        predictions.append(prediction)

    evaluation_data = evaluation_data.add_column("predicted", predictions)

    # Compute the evaluation metric
    result, test_results = eval_metric(evaluation_data, num_sequences)
    logging.info("%s", result)

    # Generate the html output for the evaluation data
    evaluation_data = evaluation_data.map(
        generate_html_output,
        batched=True,
        num_proc=4,
        fn_kwargs={"test_results": test_results},
    )

    # Save the evaluation data
    data_name = dataset_path.split("/")[-1]
    model_name = checkpoint.split("/")[-1]
    out_path = os.path.join(
        GENERATED_PATH, f"evaluation_data_{data_name}_{model_name}.data"
    )
    evaluation_data.save_to_disk(out_path)

    with open(os.path.join(out_path, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f)

    return evaluation_data, result


if __name__ == "__main__":
    args = parse_args()

    handler = logging.StreamHandler()
    logging.basicConfig(
        handlers=[handler, logging.FileHandler(LOG_PATH)],
        level=args.loglevel,
    )

    main(args)
