import os
import re
import io
import sys
import wget
import html
import pickle
import random
import tarfile
import tempfile
import itertools
import functools
import traceback
import subprocess
import multiprocessing
import concurrent.futures

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import more_itertools as more_itertools

from bs4 import BeautifulSoup
from IPython.display import display, HTML
from tqdm import tqdm
from pprint import pprint
from difflib import Differ, SequenceMatcher
from collections import Counter
from argparse import ArgumentParser

from mydifflib import group_diff_chunks, pdiff, single_change

tqdm.pandas()

input_path = "../input/"
root_path = input_path + "Project_CodeNet/"

tools_path = "../Project_CodeNet/tools/"
tokenizer_dir_path = tools_path + "spt-generator/"
spt_profile = tokenizer_dir_path + "spt.profile"
tokenizer_path = tokenizer_dir_path + "scripts/run/tokenize.sh"

data_path = root_path + "data/"
metadata_path = root_path + "metadata/"
derived_path = root_path + "derived/"
descriptions_path = root_path + "problem_descriptions/"

problem_list_clean_path = input_path + "problem_list_clean.csv"
generated_pairs_path = input_path + "generated_pairs.csv"
cleaned_generated_pairs_path = input_path + "cleaned_generated_pairs.csv"

supported_languages = ["C"]


def id2desc(problem_id):
    return descriptions_path + problem_id + ".html"


def id2inout(problem_id, name="input"):
    return derived_path + "input_output/data/" + problem_id + "/" + name + ".txt"


def id2submission(problem_id, language, submission_id, filename_ext):
    return (
        data_path
        + problem_id
        + "/"
        + language
        + "/"
        + submission_id
        + "."
        + filename_ext
    )


def read_submission_file(problem_id, language, submission_id, extension):
    """
    Read the source code as a list of lines for a given problem and submission id
    the language and extension are also required to complete the path to the file
    """
    with open(id2submission(problem_id, language, submission_id, extension)) as f:
        text = f.readlines()

    return text


data_url = "https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0"
tar_name = "Project_CodeNet.tar.gz"
tar_path = input_path + tar_name


def download_data():
    if os.path.exists(root_path):
        print("dataset root dir found")
        return

    if not os.path.exists(tar_path):
        wget.download(f"{data_url}/{tar_name}", out=tar_path)

    with tarfile.open(tar_path) as tf:
        tf.extractall(path=data_path)


def clean_problem_list(problem_list_df: pd.DataFrame = None):
    file_path = metadata_path + "problem_list.csv"
    print(f"Cleaning {file_path}")

    if problem_list_df is None:
        problem_list_df = pd.read_csv(file_path, index_col="id")

    problem_list_df["time_limit"].fillna(
        problem_list_df["time_limit"].median(), inplace=True
    )
    problem_list_df["memory_limit"].fillna(
        problem_list_df["memory_limit"].median(), inplace=True
    )

    problem_ids = problem_list_df.index.unique()

    input_mask = [os.path.exists(id2inout(problem_id)) for problem_id in problem_ids]

    problem_list_df = problem_list_df[input_mask]
    problem_ids = problem_list_df.index.unique()

    return problem_list_df


def preprocess_problem_for_language(
    problem_df: pd.DataFrame, problem_id: str, language: str = "C", extension: str = "c"
):
    submissions_diff_dfs = []

    user_ids = problem_df["user_id"].unique()
    for user_id in user_ids:
        submission_df = problem_df[problem_df["user_id"] == user_id].sort_values("date")

        if len(submission_df) < 2:
            continue

        submission_ids = submission_df.index.unique()
        for original_id, changed_id in zip(submission_ids, submission_ids[1:]):
            original_status = submission_df.loc[original_id, "status"]
            changed_status = submission_df.loc[changed_id, "status"]
            if not (original_status != "Accepted" and changed_status == "Accepted"):
                continue

            original_text = read_submission_file(
                problem_id, language, original_id, extension
            )
            changed_text = read_submission_file(
                problem_id, language, changed_id, extension
            )

            diff = pdiff(original_text, changed_text)
            diff = "".join(diff)
            chunks = group_diff_chunks(diff)

            if not (len(chunks) == 1 and single_change(chunks[0])):
                continue

            submissions_diff_dfs.append(
                (
                    original_id,
                    changed_id,
                    chunks[0][0],
                    chunks[0][2],
                    chunks[0][3],
                    original_status,
                    language,
                )
            )

    return submissions_diff_dfs


def generate_pairs_task(problem_id: str):
    columns = [
        "original_id",
        "changed_id",
        "original_line",
        "diff_op",
        "changed_line",
        "original_status",
        "original_language",
    ]
    dfs = []

    problem_df = pd.read_csv(
        metadata_path + f"{problem_id}.csv", index_col="submission_id"
    )
    if problem_df.empty:
        return

    problem_df = problem_df[
        (problem_df["status"] != "Compile Error")
        & (problem_df["language"].isin(supported_languages))
    ]
    grouped_languages = problem_df.groupby("language")

    for language, problem_df in grouped_languages:
        if problem_df.empty:
            continue
        extension = problem_df.iloc[0]["filename_ext"]
        xs = preprocess_problem_for_language(
            problem_df, problem_id, language, extension
        )
        df = pd.DataFrame(xs, columns=columns)
        df["problem_id"] = problem_id
        df["language"] = language
        df["filename_ext"] = extension
        dfs.append(df)

    return pd.DataFrame() if not dfs else pd.concat(dfs, ignore_index=True)


def generate_pairs(problem_list_df: pd.DataFrame = None):
    if problem_list_df is None:
        problem_list_df = pd.read_csv(problem_list_clean_path, index_col="id")
    dfs = []

    problem_ids = problem_list_df.index.unique()
    with tqdm(total=len(problem_ids)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            future_to_problem_id = {
                executor.submit(generate_pairs_task, problem_id): problem_id
                for problem_id in problem_ids
            }

            for future in concurrent.futures.as_completed(future_to_problem_id):
                problem_id = future_to_problem_id[future]

                try:
                    problem_pairs_df = future.result()
                    dfs.append(problem_pairs_df)
                except Exception as exc:
                    print(f"{problem_id} generated an exception: {exc}")
                    traceback.print_exc()
                else:
                    pbar.set_description(f"Processing {problem_id}")
                    pbar.update(1)

    df = pd.concat(dfs, ignore_index=True)

    return df.sort_values('original_id')


def handle_process(command, input=None, timeout=None):
    shell = False
    if not isinstance(command, list):
        shell = True

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=shell,
        encoding="utf-8",
        errors="ignore",
    )

    try:
        output, error = process.communicate(input, timeout)
        return output, error, process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise TimeoutError


def run_ctokenizer(file_path):
    grep_command = "grep -P -v '^[ \\t]*#[ ]*include[ ]*[\\<|\\\"].*(?<!\\*\\/)$'"

    with tempfile.NamedTemporaryFile("w+t", suffix=".c") as writer:
        output, error, returncode = handle_process(
            f"{grep_command} {file_path} | gcc -E -P -xc - -o {writer.name}"
        )
        assert returncode == 0, f"Error in grep and gcc {error} {output} {returncode}"
        output, error, returncode = handle_process(f"{tokenizer_path} {writer.name}")
        assert returncode == 0, f"Error in tokenize {error} {output} {returncode}"

    return pd.read_csv(io.StringIO(output), sep=",")


def run_tokenizer(problem_id, language, submission_id, filename_ext):
    if language == "C":
        return run_ctokenizer(
            id2submission(problem_id, language, submission_id, filename_ext)
        )

    assert False, "Error"


def clean_genereated_pairs_task(
    original_id,
    changed_id,
    original_line,
    diff_op,
    changed_line,
    original_status,
    original_language,
    problem_id,
    language,
    filename_ext,
):
    original_tokens_df = run_tokenizer(problem_id, language, original_id, filename_ext)
    changed_tokens_df = run_tokenizer(problem_id, language, changed_id, filename_ext)

    a = original_tokens_df["text"].values
    b = changed_tokens_df["text"].values
    s = SequenceMatcher(None, a, b)
    opcodes = [x for x in s.get_opcodes() if x[0] != "equal"]
    if len(opcodes) != 1:
        return pd.DataFrame()

    tag, i1, i2, j1, j2 = opcodes[0]
    df = original_tokens_df[i1:i2].merge(
        changed_tokens_df[j1:j2], how="outer", on=["seqnr"]
    )
    df["tag"] = tag
    df["problem_id"] = problem_id
    df["original_id"] = original_id
    df["changed_id"] = changed_id
    df["language"] = language
    df["extension"] = filename_ext

    return df


def clean_genereated_pairs(generated_pairs_df: pd.DataFrame = None):
    if generated_pairs_df is None:
        generated_pairs_df = pd.read_csv(generated_pairs_path)
    submissions_diff_dfs = []

    with tqdm(total=len(generated_pairs_df)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            future_to_problem_id = {
                executor.submit(clean_genereated_pairs_task, *row): row
                for _, row in generated_pairs_df.iterrows()
            }

            for future in concurrent.futures.as_completed(future_to_problem_id):
                (
                    original_id,
                    changed_id,
                    original_line,
                    diff_op,
                    changed_line,
                    original_status,
                    original_language,
                    problem_id,
                    language,
                    filename_ext,
                ) = future_to_problem_id[future]
                try:
                    problem_pairs_df = future.result()
                    submissions_diff_dfs.append(problem_pairs_df)
                except Exception as exc:
                    print(
                        f"{problem_id}/{language}/({original_id}|{changed_id}).{filename_ext} generated an exception: {exc}"
                    )
                    traceback.print_exc()
                else:
                    pbar.set_description(f"Processing {problem_id} {original_id}")
                    pbar.update(1)

    df = pd.concat(submissions_diff_dfs, ignore_index=True)

    return df


if __name__ == "__main__":
    os.makedirs(input_path, exist_ok=True)

    parser = ArgumentParser()
    parser.add_argument("--download", help="download the dataset", action="store_true")
    parser.add_argument(
        "--cleanlist", help="clean the problem list csv", action="store_true"
    )
    parser.add_argument(
        "--genpairs", help="generate the source code pairs", action="store_true"
    )
    parser.add_argument(
        "--cleanpairs",
        help="clean the generated source code pairs",
        action="store_true",
    )

    args = parser.parse_args()

    assert "AI4CODE_HOME" in os.environ, "You need to compile the AST Tokenizer and then source the spt.profile script\n"

    if args.download:
        download_data()
    if args.cleanlist:
        clean_problem_list().to_csv(problem_list_clean_path)
    if args.genpairs:
        generate_pairs().to_csv(generated_pairs_path, index=False)
    if args.cleanpairs:
        clean_genereated_pairs().to_csv(cleaned_generated_pairs_path, index=False)
