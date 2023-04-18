import logging
import os

import argparse
import openai
import pandas as pd
import resource
import subprocess
import time
import torch
import traceback
import uuid
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Optional

LOGGER = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(os.path.abspath(os.path.join(dir_path, os.pardir)), "input")
LOG_PATH = os.path.join(
    os.path.abspath(os.path.join(dir_path, os.pardir)), "description.log"
)
ROOT_PATH = os.path.join(INPUT_PATH, "Project_CodeNet")
DERIVED_PATH = os.path.join(ROOT_PATH, "derived")
GENERATED_PATH = os.path.join(INPUT_PATH, "bugnet")

MODEL_TEXT_DAVINCI_003 = "text-davinci-003"
MODEL_CODEGEN = "codegen"
MODEL_OPENAIGPT = "openai-gpt"

PROMPT_SIMPLE = "simple"


def make_prompt_simple(source: str) -> str:
    lines = source.splitlines()
    lines.insert(0, "```")
    lines.insert(0, "")
    lines.insert(0, "What is the bug that can happen in the given code")
    lines.append("```")

    return "\n".join(lines)


def generate_results(
    submission_pairs_df: pd.DataFrame,
    prompt_type: str = PROMPT_SIMPLE,
    model_type: str = MODEL_TEXT_DAVINCI_003,
    force: bool = False,
) -> pd.DataFrame:
    results_path = os.path.join(GENERATED_PATH, f"{model_type}_description_results.csv")

    if os.path.exists(results_path) and not force:
        LOGGER.info("%s results already generated. skiping...", model_type)
        return pd.read_csv(results_path, keep_default_na=False)

    # Cut down from the number of examples
    pairs_df = submission_pairs_df.groupby("language").head(5)

    if prompt_type == PROMPT_SIMPLE:

        def prompt_fn(row: pd.Series) -> str:
            return make_prompt_simple(row["original_src"])

    else:
        raise NotImplementedError(f"{prompt_type} is not implemented yet")

    if model_type == MODEL_TEXT_DAVINCI_003:
        assert (
            os.environ.get("OPENAI_API_KEY") is not None
        ), "You have to provide an api key for openai"

        def result_fn(prompt: str) -> str:
            response = openai.Completion.create(
                model=model_type,
                prompt=prompt,
                temperature=0.5,
                max_tokens=256,
                top_p=1,
            )
            result = response["choices"][0]["text"]
            time.sleep(15)
            return result

    elif model_type == MODEL_CODEGEN:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono").to(
            device
        )
        max_length = 512

        def result_fn(prompt: str) -> str:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            sample = model.generate(
                **inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id
            )
            result = tokenizer.decode(sample[0])
            result = result.removeprefix(prompt)
            return result

    elif model_type == MODEL_OPENAIGPT:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = pipeline("text-generation", model="openai-gpt")
        max_length = 512

        def result_fn(prompt: str) -> str:
            result = generator(prompt, max_length=max_length, num_return_sequences=1)[0][
                "generated_text"
            ]
            result = result.removeprefix(prompt)
            return result

    else:
        raise NotImplementedError(f"{model_type} is not implemented yet")

    results = []
    with tqdm(total=len(pairs_df)) as pbar:
        for pair_id, row in pairs_df.iterrows():
            try:
                prompt = prompt_fn(row)
                result = result_fn(prompt)
                results.append((pair_id, result))
            except Exception as exc:
                LOGGER.error(
                    f"{pair_id} generated an exception:"
                    + f"{exc}\ntraceback:\n{traceback.format_exc()}"
                )
            else:
                pbar.set_description(f"[generate codex] processing {pair_id}")
                pbar.update(1)

    results = pd.DataFrame(results, columns=["index", "predicted"])
    results.set_index("index", inplace=True)
    submission_pairs_df = submission_pairs_df.join(results, how="inner")
    submission_pairs_df["predicted"] = submission_pairs_df["predicted"].astype(str)

    submission_pairs_df.to_csv(results_path, index=False)

    return submission_pairs_df


if __name__ == "__main__":
    levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        dest="loglevel",
        choices=list(levels.keys()),
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    parser.add_argument(
        "-t",
        "--type",
        default=PROMPT_SIMPLE,
        choices=[PROMPT_SIMPLE],
        help="Provide the type of prompting.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=MODEL_OPENAIGPT,
        choices=[MODEL_TEXT_DAVINCI_003, MODEL_CODEGEN, MODEL_OPENAIGPT],
        help="Provide the model to use.",
    )
    args = parser.parse_args()

    prompt_type = args.type
    model_type = args.model

    LOGGER.addHandler(logging.StreamHandler())
    LOGGER.addHandler(logging.FileHandler(os.getenv("LOG_FILE", LOG_PATH)))
    LOGGER.setLevel(levels[args.loglevel])

    logging.basicConfig(
        format="%(levelname)s: %(asctime)s %(message)s",
        datefmt="%d/%m/%y %H:%M:%S",
    )

    os.makedirs(INPUT_PATH, exist_ok=True)
    os.makedirs(GENERATED_PATH, exist_ok=True)

    generated_pairs_path = os.path.join(GENERATED_PATH, "generated_pairs.csv")

    assert os.path.exists(
        generated_pairs_path
    ), f"Could not find {generated_pairs_path}. Make sure to have the BugNet dataset"

    submission_pairs_df = pd.read_csv(generated_pairs_path, keep_default_na=False)

    results_df = generate_results(
        submission_pairs_df, prompt_type=prompt_type, model_type=model_type
    )

    print(results_df)
