import logging
import os

import argparse
import openai
import pandas as pd
import time
import traceback
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

assert OPENAI_API_KEY is not None, "You have to provide an api key for openai"

dir_path = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(os.path.abspath(os.path.join(dir_path, os.pardir)), "input")
LOG_PATH = os.path.join(os.path.abspath(os.path.join(dir_path, os.pardir)), "codex.log")
GENERATED_PATH = os.path.join(INPUT_PATH, "bugnet")


def make_codex_prompt(source: str, language: str, line: int) -> str:
    if language == "C++":
        comment = "//"
    elif language == "Python":
        comment = "#"
    else:
        raise NotImplementedError(f"{language} not implemented yet")

    lines = source.splitlines()
    lines[line] = f"{lines[line]} {comment} Fixme"
    lines.append(
        f"{comment} Q: Propose a fix for the buggy line of code, using a single line of {language} code"
    )
    lines.append(f"{comment} A:")

    return "\n".join(lines)


def generate_codex_results(
    submission_pairs_df: pd.DataFrame, force: bool = False
) -> pd.DataFrame:
    codex_results_path = os.path.join(GENERATED_PATH, "codex_results.csv")

    if os.path.exists(codex_results_path) and not force:
        LOGGER.info("codex results already generated. skiping...")
        return pd.read_csv(codex_results_path, keep_default_na=False)

    # Cut down from the number of examples
    submission_pairs_df = submission_pairs_df.groupby("language").head(10)

    results = []
    with tqdm(total=len(submission_pairs_df)) as pbar:
        for pair_id, row in submission_pairs_df.iterrows():
            try:
                prompt = make_codex_prompt(
                    row["original_src"], row["language"], row["line"]
                )
                response = openai.Completion.create(
                    model="code-davinci-002",
                    prompt=prompt,
                    temperature=0.5,
                    max_tokens=256,
                    top_p=1,
                )
                codex_result = response["choices"][0]["text"]
                time.sleep(15)
                results.append((pair_id, codex_result))
            except Exception as exc:
                LOGGER.error(
                    f"{pair_id} generated an exception:"
                    + f"{exc}\ntraceback:\n{traceback.format_exc()}"
                )
            else:
                pbar.set_description(f"[generate codex] processing {pair_id}")
                pbar.update(1)

    results = pd.DataFrame(results, columns=["index", "codex_predicted"])
    results.set_index("index", inplace=True)
    submission_pairs_df = submission_pairs_df.join(results)
    submission_pairs_df["codex_predicted"] = submission_pairs_df[
        "codex_predicted"
    ].astype(str)

    submission_pairs_df.to_csv(codex_results_path, index=False)

    return submission_pairs_df


def compute_codex_accuracy(submission_pairs_df: pd.DataFrame) -> pd.DataFrame:
    line_str = submission_pairs_df.apply(
        lambda row: row["changed_src"].splitlines()[row["line"]], axis="columns"
    )

    def codex_to_line_str(codex_predicted: str) -> str:
        codex_predicted = codex_predicted.strip()
        codex_lines = codex_predicted.splitlines()
        if len(codex_lines) > 0:
            codex_predicted = codex_lines[0]
        return codex_predicted

    codex_line_str = submission_pairs_df.apply(
        lambda row: codex_to_line_str(str(row["codex_predicted"])), axis="columns"
    )

    codex_df = submission_pairs_df.copy()
    codex_df["correct"] = line_str == codex_line_str

    return codex_df


def display_codex_accuracy(codex_df: pd.DataFrame):
    codex_lang_df = codex_df.groupby("language")["correct"].agg(["sum", "count"])
    codex_lang_df["accuracy"] = codex_lang_df["sum"] / codex_lang_df["count"]

    print(codex_lang_df[["accuracy"]])


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

    codex_df = generate_codex_results(submission_pairs_df)
    codex_df = compute_codex_accuracy(codex_df)
    display_codex_accuracy(codex_df)
