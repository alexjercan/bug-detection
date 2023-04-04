import logging
import os

import argparse
import pandas as pd
import resource
import subprocess
import time
import torch
import traceback
import uuid
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

LOGGER = logging.getLogger(__name__)

MAX_VIRTUAL_MEMORY = 100 * 1024 * 1024  # 100 MB

dir_path = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(os.path.abspath(os.path.join(dir_path, os.pardir)), "input")
LOG_PATH = os.path.join(
    os.path.abspath(os.path.join(dir_path, os.pardir)), "codegen.log"
)
ROOT_PATH = os.path.join(INPUT_PATH, "Project_CodeNet")
DERIVED_PATH = os.path.join(ROOT_PATH, "derived")
GENERATED_PATH = os.path.join(INPUT_PATH, "bugnet")


def id2inout(problem_id: str, name: str = "input") -> str:
    return os.path.join(DERIVED_PATH, "input_output", "data", problem_id, f"{name}.txt")


def execute_source(
    source_code: str,
    language: str,
    input: str,
    timeout: Optional[float] = None,
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


def make_prompt_simple(source: str, language: str) -> str:
    if language == "C++":
        comment = "//"
    elif language == "Python":
        comment = "#"
    else:
        raise NotImplementedError(f"{language} not implemented yet")

    lines = source.splitlines()
    lines.append(f"{comment} Propose code to fix the bug")
    lines.append("")

    return "\n".join(lines)


def make_prompt_multishot(pairs_df: pd.DataFrame, source: str, count: int = 5) -> str:
    pairs_df = pairs_df.iloc[:count]

    result = ""
    for _, row in pairs_df.iterrows():
        result = result + row["original_src"] + "\n\n" + row["changed_src"] + "\n\n"

    result = result + source + "\n"

    return result


def generate_results(
    submission_pairs_df: pd.DataFrame, force: bool = False
) -> pd.DataFrame:
    results_path = os.path.join(GENERATED_PATH, "results.csv")

    if os.path.exists(results_path) and not force:
        LOGGER.info("codegen results already generated. skiping...")
        return pd.read_csv(results_path, keep_default_na=False)

    # Cut down from the number of examples
    pairs_df = submission_pairs_df.groupby("language").head(100)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono").to(
        device
    )

    results = []
    with tqdm(total=len(pairs_df)) as pbar:
        for pair_id, row in pairs_df.iterrows():
            try:
                prompt = make_prompt_simple(row["original_src"], row["language"])

                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                sample = model.generate(**inputs, max_length=128)
                result = tokenizer.decode(
                    sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]
                )

                with open(
                    id2inout(row["problem_id"], name="input"), "r", encoding="utf-8"
                ) as f:
                    input = f.read()
                execute_output = execute_source(
                    result, row["language"], input, timeout=2
                )

                results.append((pair_id, result, execute_output))
            except Exception as exc:
                LOGGER.error(
                    f"{pair_id} generated an exception:"
                    + f"{exc}\ntraceback:\n{traceback.format_exc()}"
                )
            else:
                pbar.set_description(f"[generate codegen] processing {pair_id}")
                pbar.update(1)

    results = pd.DataFrame(results, columns=["index", "predicted", "execute_output"])
    results.set_index("index", inplace=True)
    submission_pairs_df = submission_pairs_df.join(results, how="inner")
    submission_pairs_df["predicted"] = submission_pairs_df["predicted"].astype(str)

    submission_pairs_df.to_csv(results_path, index=False)

    return submission_pairs_df


def compute_accuracy(submission_pairs_df: pd.DataFrame) -> pd.DataFrame:
    df = submission_pairs_df.copy()
    df["correct_exact"] = df["predicted"] == df["changed_src"]

    def check(row: pd.Series) -> bool:
        with open(
            id2inout(row["problem_id"], name="output"), "r", encoding="utf-8"
        ) as f:
            output = f.read()

        return output == row["execute_output"]

    df["correct_execute"] = df.apply(check, axis="columns")

    def get_bug_type(row: pd.Series) -> str:
        j1 = row["j1"]
        j2 = row["j2"]
        line = "\n".join(row["changed_src"].splitlines()[j1:j2])
        language = row["language"]

        if language == "Python":
            return (
                "input"
                if "input" in line
                else "output"
                if "print" in line
                else "algorithm"
            )

        if language == "C++":
            return (
                "input"
                if ("cin" in line or "scanf" in line)
                else "output"
                if ("cout" in line or "printf" in line)
                else "algorithm"
            )

        raise NotImplementedError(f"{language} not implemented yet")

    df["type"] = df.apply(get_bug_type, axis="columns")

    return df


def display_accuracy_exact(df: pd.DataFrame):
    lang_df = df.groupby("language")["correct_exact"].agg(["sum", "count"])
    lang_df["accuracy"] = lang_df["sum"] / lang_df["count"]

    print("Exact match accuracy")
    print(lang_df[["accuracy"]])


def display_accuracy_execute(df: pd.DataFrame):
    lang_df = df.groupby("language")["correct_execute"].agg(["sum", "count"])
    lang_df["accuracy"] = lang_df["sum"] / lang_df["count"]

    print("Execute accuracy")
    print(lang_df[["accuracy"]])


def display_accuracy_exact_by_type(df: pd.DataFrame):
    lang_df = df.groupby(["language", "type"])["correct_exact"].agg(["sum", "count"])
    lang_df["accuracy"] = lang_df["sum"] / lang_df["count"]

    print("Exact match accuracy")
    print(lang_df[["accuracy"]])


def display_accuracy_execute_by_type(df: pd.DataFrame):
    lang_df = df.groupby(["language", "type"])["correct_execute"].agg(["sum", "count"])
    lang_df["accuracy"] = lang_df["sum"] / lang_df["count"]

    print("Execute accuracy")
    print(lang_df[["accuracy"]])


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
    args = parser.parse_args()

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

    results_df = generate_results(submission_pairs_df)
    results_df = compute_accuracy(results_df)
    display_accuracy_exact(results_df)
    display_accuracy_execute(results_df)
    display_accuracy_exact_by_type(results_df)
    display_accuracy_execute_by_type(results_df)
