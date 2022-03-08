import os
import wget
import tarfile
import traceback
import concurrent.futures

import pandas as pd

from tqdm import tqdm
from difflib import SequenceMatcher

from util import exec_file_str, extract_error_class, extract_error_class_extra
from tokenizer import (
    decode_escapes,
    generate_token_arrangements,
    run_tokenizer,
    tokens2str,
)

tqdm.pandas()

P = 8

input_path = "../input/"
root_path = input_path + "Project_CodeNet/"
generated_path = input_path + "generated/"

data_path = root_path + "data/"
generated_data_path = generated_path + "data/"
metadata_path = root_path + "metadata/"
derived_path = root_path + "derived/"
descriptions_path = root_path + "problem_descriptions/"

problem_list_clean_path = generated_path + "problem_list_clean.csv"
generated_pairs_path = generated_path + "generated_pairs.csv"
generated_opcodes_path = generated_path + "generated_opcodes.csv"
error_pairs_path = generated_path + "error_pairs.csv"

supported_languages = ["Python"]
supported_original_languages = [
    "C++14 (GCC 5.4.1)",
    "C++ (GCC 9.2.1)",
    "C++",
    "JAVA",
    # "Python (3.4.3)",
    # "PyPy3 (7.3.0)",
    "Python (3.8.2)",
    "C++11",
    # "PyPy3 (2.4.0)",
    "C",
    "C (GCC 9.2.1)",
    "C++14 (Clang 3.8.0)",
    "Python",
    "Java (OpenJDK 11.0.6)",
    "C (GCC 5.4.1)",
    # "Python (2.7.6)",
    "C++ (Clang 10.0.0)",
    "Java8 (OpenJDK 1.8.0)",
    "Python3",
    "C++ (GCC 9.2.1 with AC Library v1.1)",
    "C++14",
    "Java (OpenJDK 1.8.0)",
    "C++ (GCC 5.4.1)",
    "C (Clang 3.8.0)",
    "C (Clang 10.0.0)",
    "C++ (Clang 3.8.0)",
    "Java7 (OpenJDK 1.7.0)",
    "C++ (G++ 4.6.4)",
    "C++ (Clang 10.0.0 with AC Library v1.1)",
    # "PyPy2 (5.6.0)",
    "C++11 (GCC 4.8.1)",
    # "PyPy2 (7.3.0)",
    # "Python (3.4.2)",
]

data_url = "https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0"
tar_name = "Project_CodeNet.tar.gz"
tar_path = input_path + tar_name


def id2desc(problem_id: str) -> str:
    return descriptions_path + problem_id + ".html"


def id2inout(problem_id: str, name: str = "input") -> str:
    return derived_path + "input_output/data/" + problem_id + "/" + name + ".txt"


def id2submission(
    problem_id: str,
    language: str,
    submission_id: str,
    filename_ext: str,
    data_path: str = data_path,
) -> str:
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


def read_submission_file(
    problem_id: str,
    language: str,
    submission_id: str,
    extension: str,
    data_path: str = data_path,
) -> list[str]:
    """
    Read the source code as a list of lines for a given problem and submission id
    the language and extension are also required to complete the path to the file
    """
    with open(
        id2submission(problem_id, language, submission_id, extension, data_path)
    ) as f:
        text = f.readlines()

    return text


def download_codenet(force: bool = False) -> None:
    if os.path.exists(root_path) and not force:
        print("Dataset root dir found. skiping...")
        return

    if not os.path.exists(tar_path) or force:
        wget.download(f"{data_url}/{tar_name}", out=tar_path)

    with tarfile.open(tar_path) as tf:
        tf.extractall(path=data_path)


def clean_codenet(force: bool = False):
    if os.path.exists(problem_list_clean_path) and not force:
        print("Dataset was already cleaned. skiping...")
        return

    file_path = metadata_path + "problem_list.csv"
    print(f"Cleaning {file_path}")

    problem_list_df = pd.read_csv(file_path, index_col="id")

    problem_list_df["time_limit"].fillna(
        problem_list_df["time_limit"].median(), inplace=True
    )
    problem_list_df["memory_limit"].fillna(
        problem_list_df["memory_limit"].median(), inplace=True
    )

    problem_ids = problem_list_df.index.unique()

    input_mask = [
        os.path.exists(id2inout(str(problem_id))) for problem_id in problem_ids
    ]

    problem_list_df = problem_list_df.loc[input_mask]
    problem_ids = problem_list_df.index.unique()

    problem_list_df.to_csv(problem_list_clean_path)


def generate_pairs_task(problem_id: str) -> pd.DataFrame:
    columns = [
        "original_id",
        "changed_id",
        "original_status",
    ]
    dfs = []

    problem_df = pd.read_csv(
        metadata_path + f"{problem_id}.csv", index_col="submission_id"
    )
    if problem_df.empty:
        return pd.DataFrame()

    problem_df = problem_df[
        (problem_df["status"] != "Compile Error")
        & (problem_df["status"] != "Wrong Answer")
        & (problem_df["language"].isin(supported_languages))
        & (problem_df["original_language"].isin(supported_original_languages))
    ]
    grouped_languages = problem_df.groupby("language")

    for language, problem_df in grouped_languages:
        if problem_df.empty:
            continue

        submissions_diff_dfs = []

        user_ids = problem_df["user_id"].unique()
        for user_id in user_ids:
            submission_df = problem_df[problem_df["user_id"] == user_id].sort_values(
                "date"
            )

            if len(submission_df) < 2:
                continue

            submission_ids = submission_df.index.unique()
            for original_id, changed_id in zip(submission_ids, submission_ids[1:]):
                original_status = submission_df.loc[original_id, "status"]
                changed_status = submission_df.loc[changed_id, "status"]
                if not (original_status != "Accepted" and changed_status == "Accepted"):
                    continue

                submissions_diff_dfs.append(
                    (
                        original_id,
                        changed_id,
                        original_status,
                    )
                )

        df = pd.DataFrame(submissions_diff_dfs, columns=columns)
        df["problem_id"] = problem_id
        df["language"] = language
        df["filename_ext"] = problem_df.iloc[0]["filename_ext"]
        dfs.append(df)

    return pd.DataFrame() if not dfs else pd.concat(dfs, ignore_index=True)


def generate_pairs_codenet(force: bool = False):
    if os.path.exists(generated_pairs_path) and not force:
        print("Pairs already generated. skiping...")
        return

    problem_list_df = pd.read_csv(problem_list_clean_path, index_col="id")
    dfs = []

    problem_ids = problem_list_df.index.unique()
    with tqdm(total=len(problem_ids)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=P) as executor:
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
    df.sort_values("original_id").to_csv(generated_pairs_path, index=False)


def tokenize_pairs_task(
    original_id: str,
    changed_id: str,
    original_status: str,
    problem_id: str,
    language: str,
    filename_ext: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        run_tokenizer(
            id2submission(problem_id, language, original_id, filename_ext), language
        ),
        run_tokenizer(
            id2submission(problem_id, language, changed_id, filename_ext), language
        ),
    )


def tokenize_pairs_codenet(force: bool = False):
    if os.path.exists(generated_data_path) and not force:
        print("Tokens already generated. skiping...")
        return

    df = pd.read_csv(generated_pairs_path)

    with tqdm(total=len(df)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=P) as executor:
            future_to_problem_id = {
                executor.submit(tokenize_pairs_task, *row): row
                for _, row in df.iterrows()
            }

            for future in concurrent.futures.as_completed(future_to_problem_id):
                (
                    original_id,
                    changed_id,
                    original_status,
                    problem_id,
                    language,
                    filename_ext,
                ) = future_to_problem_id[future]
                try:
                    original_tokenized_df, changed_tokenized_df = future.result()
                    original_tokenized_path = id2submission(
                        problem_id, language, original_id, "csv", generated_data_path
                    )
                    changed_tokenized_path = id2submission(
                        problem_id, language, changed_id, "csv", generated_data_path
                    )
                    os.makedirs(os.path.dirname(original_tokenized_path), exist_ok=True)
                    original_tokenized_df.to_csv(original_tokenized_path, index=False)
                    os.makedirs(os.path.dirname(changed_tokenized_path), exist_ok=True)
                    changed_tokenized_df.to_csv(changed_tokenized_path, index=False)
                except Exception as exc:
                    print(
                        f"{problem_id}/{language}/({original_id}|{changed_id}).{filename_ext} generated an exception: {exc}"
                    )
                    traceback.print_exc()
                else:
                    pbar.set_description(f"Processing {problem_id} {original_id}")
                    pbar.update(1)


def generate_opcodes_task(
    original_id: str,
    changed_id: str,
    original_status: str,
    problem_id: str,
    language: str,
    filename_ext: str,
) -> pd.DataFrame:
    original_tokens_df = pd.read_csv(
        id2submission(problem_id, language, original_id, "csv", generated_data_path),
        keep_default_na=False,
        index_col="seqnr",
    )
    changed_tokens_df = pd.read_csv(
        id2submission(problem_id, language, changed_id, "csv", generated_data_path),
        keep_default_na=False,
        index_col="seqnr",
    )

    a = original_tokens_df["text"].values.tolist()
    b = changed_tokens_df["text"].values.tolist()
    s: SequenceMatcher = SequenceMatcher(None, a, b)
    opcodes = [x for x in s.get_opcodes() if x[0] != "equal"]

    opcodes_df = pd.DataFrame(dict(zip(["tag", "i1", "i2", "j1", "j2"], zip(*opcodes))))
    opcodes_df["original_id"] = original_id
    opcodes_df["changed_id"] = changed_id
    opcodes_df["problem_id"] = problem_id

    return opcodes_df


def generate_opcodes_codenet(force: bool = False) -> pd.DataFrame:
    if os.path.exists(generated_opcodes_path) and not force:
        print("Opcodes already generated. skiping...")
        return

    generated_pairs_df = pd.read_csv(generated_pairs_path)
    opcodes_dfs = []

    with tqdm(total=len(generated_pairs_df)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=P) as executor:
            future_to_problem_id = {
                executor.submit(generate_opcodes_task, *row): row
                for _, row in generated_pairs_df.iterrows()
            }

            for future in concurrent.futures.as_completed(future_to_problem_id):
                (
                    original_id,
                    changed_id,
                    original_status,
                    problem_id,
                    language,
                    filename_ext,
                ) = future_to_problem_id[future]
                try:
                    opcodes_df = future.result()
                    opcodes_dfs.append(opcodes_df)
                except Exception as exc:
                    print(
                        f"{problem_id}/{language}/({original_id}|{changed_id}).{filename_ext} generated an exception: {exc}"
                    )
                    traceback.print_exc()
                else:
                    pbar.set_description(f"Processing {problem_id} {original_id}")
                    pbar.update(1)

    pd.concat(opcodes_dfs).to_csv(generated_opcodes_path, index=False)


def add_error_description_task(
    time_limit: float,
    problem_id: str,
    original_id: str,
    changed_id: str,
    language: str,
    filename_ext: str,
    opcodes_df: pd.DataFrame,
) -> list[tuple[int, str, str, int, str, str]]:
    csv_path = id2submission(
        problem_id, language, original_id, "csv", generated_data_path
    )
    original_df = pd.read_csv(csv_path, keep_default_na=False)

    csv_path = id2submission(
        problem_id, language, changed_id, "csv", generated_data_path
    )
    changed_df = pd.read_csv(csv_path, keep_default_na=False)

    input_path = id2inout(problem_id, name="input")
    output_path = id2inout(problem_id, name="output")

    with open(input_path, "r") as f:
        input = f.read()

    timeout = time_limit / 1000 * 1.5

    errs = []
    variant_dfs = generate_token_arrangements(original_df, changed_df, opcodes_df)
    for _id, variant_df in variant_dfs:
        tokens_df = variant_df[["text", "class"]]

        try:
            source_code = decode_escapes(tokens2str(tokens_df, language))

            output, error, returncode = exec_file_str(
                source_code, input, timeout, language
            )
        except (AssertionError, DeprecationWarning) as exc:
            output = ""
            returncode = 1
            error = str(exc)

        error_class = extract_error_class((language, error, returncode))
        error_class_extra = extract_error_class_extra((language, error, returncode))

        errs.append((_id, output, error, returncode, error_class, error_class_extra))

    return errs


def add_error_description_codenet(force: bool = False) -> pd.DataFrame:
    if os.path.exists(error_pairs_path) and not force:
        print("Opcodes already generated. skiping...")
        return

    generated_pairs_df = pd.read_csv(generated_pairs_path)
    problem_list_df = pd.read_csv(problem_list_clean_path, index_col="id")
    generated_opcodes_df = pd.read_csv(generated_opcodes_path)
    generated_opcodes_df = generated_opcodes_df.astype(
        {"i1": int, "i2": int, "j1": int, "j2": int}
    )

    time_limit_f = lambda pid: problem_list_df.loc[pid]["time_limit"]

    df = generated_opcodes_df.merge(
        generated_pairs_df, on=["original_id", "changed_id", "problem_id"]
    )
    gs = df.groupby(
        ["problem_id", "original_id", "changed_id", "language", "filename_ext"]
    ).groups

    errs = []
    with tqdm(total=len(gs)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=P) as executor:
            future_to_problem_id = {
                executor.submit(
                    add_error_description_task,
                    time_limit_f(key[0]),
                    *key,
                    df.iloc[_ids][["tag", "i1", "i2", "j1", "j2"]],
                ): key
                for key, _ids in gs.items()
            }

            for future in concurrent.futures.as_completed(future_to_problem_id):
                (
                    problem_id,
                    original_id,
                    changed_id,
                    language,
                    filename_ext,
                ) = future_to_problem_id[future]
                try:
                    err = future.result()
                    errs.extend(err)
                except Exception as exc:
                    print(
                        f"{problem_id}/{language}/({original_id}|{changed_id}).{filename_ext} generated an exception: {exc}"
                    )
                    traceback.print_exc()
                else:
                    pbar.set_description(f"Processing {problem_id} {original_id}")
                    pbar.update(1)

    errs_df = pd.DataFrame(
        errs,
        columns=[
            "index",
            "output",
            "error",
            "returncode",
            "error_class",
            "error_class_extra",
        ],
    ).set_index("index")
    errs_df.index.name = None

    df = generated_opcodes_df.join(errs_df).sort_index()
    df = df[df["returncode"] != 0].dropna(subset=["error_class"])
    df = generated_pairs_df.merge(df)

    df.to_csv(error_pairs_path, index=False)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(generated_path), exist_ok=True)

    download_codenet()
    clean_codenet()
    generate_pairs_codenet()
    tokenize_pairs_codenet()
    generate_opcodes_codenet()
    add_error_description_codenet()
