import logging
import os

import argparse
import openai
import pandas as pd
import pickle
import time
import torch
import traceback
from pathlib import Path
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
)
from typing import List, Tuple

LOGGER = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(os.path.abspath(os.path.join(dir_path, os.pardir)), "input")
LOG_PATH = os.path.join(
    os.path.abspath(os.path.join(dir_path, os.pardir)), "description.log"
)
ROOT_PATH = os.path.join(INPUT_PATH, "Project_CodeNet")
DERIVED_PATH = os.path.join(ROOT_PATH, "derived")
GENERATED_PATH = os.path.join(INPUT_PATH, "bugnet")

MODEL_CHATGPT = "gpt-3.5-turbo"
MODEL_TEXT_DAVINCI_003 = "text-davinci-003"
MODEL_CODEGEN = "codegen"
MODEL_OPENAIGPT = "openai-gpt"
MODEL_GPT2 = "gpt2"

PROMPT_SIMPLE = "simple"
PROMPT_MULTISHOT = "multishot"


def make_prompt_simple(source: str) -> str:
    lines = source.splitlines()
    lines.insert(0, "```")
    lines.insert(0, "")
    lines.insert(0, "What is the bug that can happen in the given code")
    lines.append("```")

    return "\n".join(lines)


def make_prompt_multishot(
    pairs_df: pd.DataFrame, source_id: int, source_row: pd.Series, count: int = 5
) -> str:
    pairs_df = pairs_df[
        (pairs_df["language"] == source_row["language"]) & (pairs_df.index != source_id)
    ]
    pairs_df = pairs_df.iloc[:count]

    result = ""
    for _, row in pairs_df.iterrows():
        result = result + row["original_src"] + "\n\n" + row["error"] + "\n\n"

    result = result + source_row["original_src"] + "\n"

    return result


def generate_results(
    submission_pairs_df: pd.DataFrame,
    prompt_type: str = PROMPT_SIMPLE,
    model_type: str = MODEL_CHATGPT,
    force: bool = False,
) -> Tuple[pd.DataFrame, List[List[torch.Tensor]]]:
    results_path = os.path.join(GENERATED_PATH, f"{model_type}_description_results.csv")

    if os.path.exists(results_path) and not force:
        LOGGER.info("%s results already generated. skiping...", model_type)
        return pd.read_csv(results_path, keep_default_na=False)

    # Cut down from the number of examples
    pairs_df = submission_pairs_df.groupby("language").head(1)

    if prompt_type == PROMPT_SIMPLE:

        def prompt_fn(_: int, row: pd.Series) -> str:
            return make_prompt_simple(row["original_src"])

    elif prompt_type == PROMPT_MULTISHOT:

        def prompt_fn(row_id: int, row: pd.Series) -> str:
            return make_prompt_multishot(pairs_df, row_id, row, count=5)

    else:
        raise NotImplementedError(f"{prompt_type} is not implemented yet")

    if model_type == MODEL_TEXT_DAVINCI_003:
        assert (
            os.environ.get("OPENAI_API_KEY") is not None
        ), "You have to provide an api key for openai"

        def result_fn(prompt: str) -> Tuple[str, List[torch.Tensor]]:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a senior developer."},
                    {"role": "user", "content": prompt},
                ]
            )
            result = response["choices"][0]["message"]["content"]
            time.sleep(1)
            return result, None

    elif model_type == MODEL_TEXT_DAVINCI_003:
        assert (
            os.environ.get("OPENAI_API_KEY") is not None
        ), "You have to provide an api key for openai"

        def result_fn(prompt: str) -> Tuple[str, List[torch.Tensor]]:
            response = openai.Completion.create(
                model=model_type,
                prompt=prompt,
                temperature=0.5,
                max_tokens=256,
                top_p=1,
            )
            result = response["choices"][0]["text"]
            time.sleep(15)
            return result, None

    elif model_type == MODEL_CODEGEN:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono").to(
            device
        )

        def result_fn(prompt: str) -> Tuple[str, List[torch.Tensor]]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            sample = model(**inputs, output_attentions=True)
            preds = sample.logits.argmax(dim=2)
            result = tokenizer.decode(preds[0])
            result = result.removeprefix(prompt)
            return result, sample.attentions

    elif model_type == MODEL_OPENAIGPT:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
        model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")

        def result_fn(prompt: str) -> Tuple[str, List[torch.Tensor]]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            sample = model(**inputs, output_attentions=True)
            preds = sample.logits.argmax(dim=2)
            result = tokenizer.decode(preds[0])
            result = result.removeprefix(prompt)
            return result, sample.attentions

    elif model_type == MODEL_GPT2:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

        def result_fn(prompt: str) -> Tuple[str, List[torch.Tensor]]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            sample = model(**inputs, output_attentions=True)
            preds = sample.logits.argmax(dim=2)
            result = tokenizer.decode(preds[0])
            result = result.removeprefix(prompt)
            return result, sample.attentions

    else:
        raise NotImplementedError(f"{model_type} is not implemented yet")

    results = []
    attentions_results = []
    with tqdm(total=len(pairs_df)) as pbar:
        for pair_id, row in pairs_df.iterrows():
            try:
                prompt = prompt_fn(pair_id, row)
                result, attentions = result_fn(prompt)
                results.append((pair_id, result))
                attentions_results.append(attentions)
            except Exception as exc:
                LOGGER.error(
                    f"{pair_id} generated an exception:"
                    + f"{exc}\ntraceback:\n{traceback.format_exc()}"
                )
            else:
                pbar.set_description(f"[generate description] processing {pair_id}")
                pbar.update(1)

    results = pd.DataFrame(results, columns=["index", "predicted"])
    results.set_index("index", inplace=True)
    submission_pairs_df = submission_pairs_df.join(results, how="inner")
    submission_pairs_df["predicted"] = submission_pairs_df["predicted"].astype(str)

    submission_pairs_df.to_csv(results_path, index=False)

    results_path = Path(results_path).with_suffix(".pkl")
    with open(results_path, "wb") as f:
        pickle.dump(attentions_results, f)

    return submission_pairs_df, attentions_results


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
        choices=[PROMPT_SIMPLE, PROMPT_MULTISHOT],
        help="Provide the type of prompting.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=MODEL_CHATGPT,
        choices=[MODEL_CHATGPT, MODEL_TEXT_DAVINCI_003, MODEL_CODEGEN, MODEL_OPENAIGPT, MODEL_GPT2],
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

    results_df, _ = generate_results(
        submission_pairs_df, prompt_type=prompt_type, model_type=model_type
    )

    print(results_df)
