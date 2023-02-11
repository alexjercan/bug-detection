import os
import re
import wget
import json
import tarfile
import functools
import traceback
import subprocess
import concurrent.futures

import pandas as pd

from tqdm import tqdm
from typing import Union
from zipfile import ZipFile

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
error_pairs_path = generated_path + "error_pairs.csv"
codenetpy_path = generated_path + "codenetpy.json"
codenetpy_train_path = generated_path + "codenetpy_train.json"
codenetpy_test_path = generated_path + "codenetpy_test.json"
filter_problem_statements_path = generated_path + "problem_descriptions/"
kaggle_zip_path = generated_path + "kaggle.zip"

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


def handle_process(
    command: Union[str, list[str]], input: str = None, timeout: float = None
) -> tuple[str, str, int]:
    shell = not isinstance(command, list)

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
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        output, error = "", "TLEError: Time limit exceeded"

    return output, error, process.returncode


def extract_error_class_python(error: str, returncode: int) -> str:
    rs = "|".join(
        [
            r"^(\w*Error):.*",
            r"(\w*Warning):.*",
        ]
    )

    p_class = re.compile(rs, re.MULTILINE)
    error_class = p_class.findall(error)
    if not error_class:
        return str(returncode)
    return functools.reduce(lambda acc, x: acc or x, error_class[0], None)


def extract_error_class_extra_python(error: str, returncode: int) -> str:
    rs = "|".join(
        [
            r"^(\w*Error:.*).*",
            r"(\w*Warning:.*).*",
        ]
    )

    p_class_extra = re.compile(rs, re.MULTILINE)
    error_class_extra = p_class_extra.findall(error)
    if not error_class_extra:
        return error
    return functools.reduce(lambda acc, x: acc or x, error_class_extra[0], None)


def extract_error_class_c(error: str, returncode: int) -> str:
    return str(returncode)


def extract_error_class_extra_c(error: str, returncode: int) -> str:
    rs = "|".join(
        [
            r"(undefined reference .*)",
            r"(\*\*\* stack smashing detected \*\*\*: terminated)",
            r"(\*\*\* buffer overflow detected \*\*\*: terminated)",
            r"(munmap_chunk\(\): .*)",
            r"(segmentation fault \(core dumped\))",
            r"(error: .*)",
            r"(relocation truncated to fit: .*)",
            r"(sysmalloc: .*)",
            r"(malloc\(\): .*)",
            r"(free\(\): .*)",
        ]
    )
    p_class_extra = re.compile(rs, re.MULTILINE)
    error_class_extra = p_class_extra.findall(error)
    if not error_class_extra:
        return error
    return functools.reduce(lambda acc, x: acc or x, error_class_extra[0], None)


def extract_error_class_java(error: str, returncode: int) -> str:
    rs = r"Exception in thread \".*?\" ([^:\n]*)"

    p_class = re.compile(rs, re.MULTILINE)
    error_class = p_class.findall(error)
    if not error_class:
        return error
    return error_class[0]


def extract_error_class_extra_java(error: str, returncode: int) -> str:
    rs = r"(Exception .*)"

    p_class_extra = re.compile(rs, re.MULTILINE)
    error_class_extra = p_class_extra.findall(error)
    if not error_class_extra:
        return error
    return error_class_extra[0]


def extract_error_class(row: pd.Series) -> str:
    language, error, returncode = row
    if language == "C":
        return extract_error_class_c(error, returncode)
    if language == "Python":
        return extract_error_class_python(error, returncode)
    if language == "C++":
        return extract_error_class_c(error, returncode)
    if language == "Java":
        return extract_error_class_java(error, returncode)

    return ""


def extract_error_class_extra(row: pd.Series) -> str:
    language, error, returncode = row
    if language == "C":
        return extract_error_class_extra_c(error, returncode)
    if language == "Python":
        return extract_error_class_extra_python(error, returncode)
    if language == "C++":
        return extract_error_class_extra_c(error, returncode)
    if language == "Java":
        return extract_error_class_extra_java(error, returncode)

    return ""


def exec_file_python(
    file_path: str, input: str = None, timeout: float = 2.0
) -> tuple[str, str, int]:
    return handle_process(["python3", file_path], input, timeout)


def exec_file(
    file_path: str, input: str = None, timeout: float = 2.0, language: str = None
) -> tuple[str, str, int]:
    if language == "Python":
        return exec_file_python(file_path, input, timeout)
    raise NotImplementedError


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
                    pbar.set_description(f"[Generate Pairs] Processing {problem_id}")
                    pbar.update(1)

    df = pd.concat(dfs, ignore_index=True)
    df.sort_values("original_id").to_csv(generated_pairs_path, index=False)


def generate_error_description_task(
    time_limit: float,
    original_id: str,
    changed_id: str,
    original_status: str,
    problem_id: str,
    language: str,
    filename_ext: str,
) -> dict:
    source_code_path = id2submission(problem_id, language, original_id, filename_ext)

    input_path = id2inout(problem_id, name="input")
    with open(input_path, "r") as f:
        input = f.read()

    timeout = time_limit / 1000 * 1.5

    try:
        output, error, returncode = exec_file(
            source_code_path, input, timeout, language
        )
    except (AssertionError, DeprecationWarning) as exc:
        output = ""
        returncode = 1
        error = str(exc)

    error_class = extract_error_class((language, error, returncode))
    error_class_extra = extract_error_class_extra((language, error, returncode))

    return {
        "problem_id": problem_id,
        "original_id": original_id,
        "changed_id": changed_id,
        "language": language,
        "filename_ext": filename_ext,
        "original_status": original_status,
        "returncode": returncode,
        "error_class": error_class,
        "error_class_extra": error_class_extra,
        "error": error,
        "output": output,
    }


def generate_error_description_codenet(force: bool = False) -> pd.DataFrame:
    if os.path.exists(error_pairs_path) and not force:
        print("Error Descriptions already generated. skiping...")
        return

    generated_pairs_df = pd.read_csv(generated_pairs_path)
    problem_list_df = pd.read_csv(problem_list_clean_path, index_col="id")

    time_limit_f = lambda pid: problem_list_df.loc[pid]["time_limit"]

    errs = []
    with tqdm(total=len(generated_pairs_df)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=P) as executor:
            future_to_problem_id = {
                executor.submit(
                    generate_error_description_task,
                    time_limit_f(row["problem_id"]),
                    *row,
                ): row
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
                    err = future.result()
                    errs.append(err)
                except Exception as exc:
                    print(
                        f"{problem_id}/{language}/({original_id}|{changed_id}).{filename_ext} generated an exception: {exc}"
                    )
                    traceback.print_exc()
                else:
                    pbar.set_description(
                        f"[Generate Error] Processing {problem_id} {original_id}"
                    )
                    pbar.update(1)

    errs_df = pd.DataFrame(errs)
    errs_df.to_csv(error_pairs_path, index=False)


def generate_labels_task(
    problem_id: str,
    original_id: str,
    changed_id: str,
    language: str,
    filename_ext: str,
    original_status: str,
    returncode: int,
    error_class: str,
    error_class_extra: str,
    error: str,
    output: str,
) -> dict:
    original_src = "".join(
        read_submission_file(problem_id, language, original_id, filename_ext)
    )
    changed_src = "".join(
        read_submission_file(problem_id, language, changed_id, filename_ext)
    )

    return {
        "original_src": original_src,
        "changed_src": changed_src,
        "problem_id": problem_id,
        "original_id": original_id,
        "changed_id": changed_id,
        "language": language,
        "filename_ext": filename_ext,
        "original_status": original_status,
        "returncode": returncode,
        "error_class": error_class,
        "error_class_extra": error_class_extra,
        "error": error,
        "output": output,
    }


def generate_labels_codenet(force: bool = False):
    if os.path.exists(codenetpy_path) and not force:
        print("Labels already generated. skiping...")
        return

    errs_df = pd.read_csv(error_pairs_path, keep_default_na=False)

    labels = []
    with tqdm(total=len(errs_df)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=P) as executor:
            future_to_problem_id = {
                executor.submit(generate_labels_task, *row,): row[
                    [
                        "original_id",
                        "changed_id",
                        "original_status",
                        "problem_id",
                        "language",
                        "filename_ext",
                    ]
                ]
                for _, row in errs_df.iterrows()
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
                    label = future.result()
                    labels.append(label)
                except Exception as exc:
                    print(
                        f"{problem_id}/{language}/({original_id}|{changed_id}).{filename_ext} generated an exception: {exc}"
                    )
                    traceback.print_exc()
                else:
                    pbar.set_description(
                        f"[Generate Labels] Processing {problem_id} {original_id}"
                    )
                    pbar.update(1)

    with open(codenetpy_path, "w") as f:
        json.dump(labels, f)


def generate_train_test_splits(force: bool = False):
    if (
        os.path.exists(codenetpy_train_path)
        and os.path.exists(codenetpy_test_path)
        and not force
    ):
        print("Train and Test splits already generated. skiping...")
        return

    with open(codenetpy_path, "r") as f:
        labels = json.load(f)

    labels_df = pd.DataFrame(labels)
    labels_df.sort_values(by=["problem_id"], inplace=True)

    train_df = labels_df.head(int(len(labels) * 0.8))
    test_df = labels_df.tail(len(labels_df) - int(len(labels_df) * 0.8))

    with open(codenetpy_train_path, "w") as f:
        json.dump({"data": train_df.to_dict(orient="records")}, f)

    with open(codenetpy_test_path, "w") as f:
        json.dump({"data": test_df.to_dict(orient="records")}, f)


def filter_problem_statements(force: bool = False):
    if os.path.exists(filter_problem_statements_path) and not force:
        print("Problem Statements already filtered. skiping...")
        return

    os.makedirs(filter_problem_statements_path, exist_ok=True)

    with open(codenetpy_path, "r") as f:
        labels = json.load(f)

    labels_df = pd.DataFrame(labels)
    labels_df.sort_values(by=["problem_id"], inplace=True)

    for problem_id in tqdm(labels_df["problem_id"]):
        src = id2desc(problem_id)
        dst = os.path.join(filter_problem_statements_path, os.path.basename(src))
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())


def prepare_kaggle():
    with ZipFile(kaggle_zip_path, "w") as zip_obj:
        zip_obj.write(codenetpy_train_path, os.path.basename(codenetpy_train_path))
        zip_obj.write(codenetpy_test_path, os.path.basename(codenetpy_test_path))
        for folder_path, _, filenames in os.walk(filter_problem_statements_path):
            for filename in filenames:
                filePath = os.path.join(folder_path, filename)
                zip_obj.write(
                    filePath,
                    os.path.join(
                        os.path.basename(os.path.normpath(folder_path)), filename
                    ),
                )


if __name__ == "__main__":
    os.makedirs(os.path.dirname(generated_path), exist_ok=True)

    download_codenet()
    clean_codenet()
    generate_pairs_codenet()
    generate_error_description_codenet()
    generate_labels_codenet()
    generate_train_test_splits()
    filter_problem_statements()
    prepare_kaggle()
