import functools
import logging
import os

import argparse
import concurrent.futures
import glob
import pandas as pd
import re
import resource
import subprocess
import traceback
import uuid
import wget
from difflib import SequenceMatcher
from multiprocessing import cpu_count
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional, Tuple
from zipfile import ZipFile

LOGGER = logging.getLogger(__name__)

MAX_VIRTUAL_MEMORY = 100 * 1024 * 1024  # 100 MB
P = int(os.environ.get("CODENET_CPUS", cpu_count()))

SUPPORTED_LANGUAGES = ["C++", "Python"]
EXTENSIONS = {"C++": "cpp", "Python": "py"}

SUPPORTED_ERRORS = [
    "Accepted",
    "Runtime Error",
    "Time Limit Exceeded",
    "Memory Limit Exceeded",
]

assert set(SUPPORTED_LANGUAGES).issubset(set(EXTENSIONS)), (
    "Expected to have extension for all supported languages, "
    "but the following languages are missing: "
    f"{', '.join(set(SUPPORTED_LANGUAGES).difference(set(EXTENSIONS)))}"
)

dir_path = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(os.path.abspath(os.path.join(dir_path, os.pardir)), "input")
LOG_PATH = os.path.join(os.path.abspath(os.path.join(dir_path, os.pardir)), "bugnet.log")
ROOT_PATH = os.path.join(INPUT_PATH, "Project_CodeNet")
DATA_PATH = os.path.join(ROOT_PATH, "data")
META_PATH = os.path.join(ROOT_PATH, "metadata")
DERIVED_PATH = os.path.join(ROOT_PATH, "derived")
PROBLEM_DESCRIPTIONS_PATH = os.path.join(ROOT_PATH, "problem_descriptions")
GENERATED_PATH = os.path.join(INPUT_PATH, "bugnet")


def id2inout(problem_id: str, name: str = "input") -> str:
    return os.path.join(DERIVED_PATH, "input_output", "data", problem_id, f"{name}.txt")


def id2desc(problem_id: str) -> str:
    return os.path.join(PROBLEM_DESCRIPTIONS_PATH, problem_id + ".html")


def id2submission(
    problem_id: str,
    language: str,
    submission_id: str,
) -> str:
    return os.path.join(
        DATA_PATH,
        problem_id,
        language,
        f"{submission_id}.{EXTENSIONS[language]}",
    )


def read_format_submission(
    problem_id: str, language: str, submission_id: str
) -> Tuple[bool, str]:
    path = id2submission(problem_id, language, submission_id)

    if language == "C++":
        try:
            result = subprocess.run(
                ["clang-format", path],
                capture_output=True,
                encoding="utf-8",
                errors="ignore",
                preexec_fn=lambda: resource.setrlimit(
                    resource.RLIMIT_AS,
                    (1024 * 1024 * 1024, resource.RLIM_INFINITY),
                ),
                check=True,
            )
            return result.returncode == 0, result.stdout
        except subprocess.CalledProcessError:
            return False, ""

    if language == "Python":
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        try:
            result = subprocess.run(
                ["black", "-q", "-"],
                input=content,
                capture_output=True,
                encoding="utf-8",
                errors="ignore",
                preexec_fn=lambda: resource.setrlimit(
                    resource.RLIMIT_AS,
                    (1024 * 1024 * 1024, resource.RLIM_INFINITY),
                ),
                check=True,
            )
            return result.returncode == 0, result.stdout
        except subprocess.CalledProcessError:
            return False, ""

    raise NotImplementedError(f"{language} not implemented yet")


def line_diff_checker(
    original_src: str, changed_src: str
) -> Optional[Tuple[str, int, int, int, int]]:
    original_src_lines = original_src.splitlines()
    changed_src_lines = changed_src.splitlines()

    opcodes = SequenceMatcher(None, original_src_lines, changed_src_lines).get_opcodes()
    changes = list(filter(lambda opcode: opcode[0] != "equal", opcodes))
    if len(changes) != 1:
        return None

    _, i1, i2, j1, j2 = changes[0]
    if i2 - i1 > 1 or j2 - j1 > 1:
        return None

    return changes[0]


def chunk_diff_checker(
    original_src: str, changed_src: str
) -> Optional[Tuple[str, int, int, int, int]]:
    opcodes = SequenceMatcher(None, original_src, changed_src).get_opcodes()
    changes = list(filter(lambda opcode: opcode[0] != "equal", opcodes))
    if len(changes) != 1:
        return None

    return changes[0]


def lines_diff_checker(
    original_src: str, changed_src: str
) -> Optional[Tuple[str, int, int, int, int]]:
    original_src_lines = original_src.splitlines()
    changed_src_lines = changed_src.splitlines()

    opcodes = SequenceMatcher(None, original_src_lines, changed_src_lines).get_opcodes()
    changes = list(filter(lambda opcode: opcode[0] != "equal", opcodes))
    if len(changes) != 1:
        return None

    return changes[0]


def lines_diff_chunk_checker(
    original_src: str, changed_src: str
) -> Optional[Tuple[str, int, int, int, int]]:
    original_src_lines = original_src.splitlines()
    changed_src_lines = changed_src.splitlines()

    opcodes = SequenceMatcher(None, original_src_lines, changed_src_lines).get_opcodes()
    changes = list(filter(lambda opcode: opcode[0] != "equal", opcodes))
    if not changes:
        return None

    if len(changes) == 1:
        return changes[0]

    _, i1, _, j1, _ = changes[0]
    _, _, i2, _, j2 = changes[-1]

    return "replace", i1, i2, j1, j2


def linter_check_submission(problem_id: str, language: str, submission_id: str) -> bool:
    path = id2submission(problem_id, language, submission_id)

    if language == "C++":
        try:
            result = subprocess.run(
                ["clang-tidy", path],
                capture_output=True,
                encoding="utf-8",
                errors="ignore",
                preexec_fn=lambda: resource.setrlimit(
                    resource.RLIMIT_AS,
                    (1024 * 1024 * 1024, resource.RLIM_INFINITY),
                ),
                check=True,
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False

    if language == "Python":
        try:
            result = subprocess.run(
                ["flake8", path],
                capture_output=True,
                encoding="utf-8",
                errors="ignore",
                preexec_fn=lambda: resource.setrlimit(
                    resource.RLIMIT_AS,
                    (1024 * 1024 * 1024, resource.RLIM_INFINITY),
                ),
                check=True,
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False

    raise NotImplementedError(f"{language} not implemented yet")


def execute_source(
    problem_id: str,
    language: str,
    submission_id: str,
    input_: str,
    output: str,
    timeout: Optional[float] = None,
) -> Optional[Tuple[str, str, str]]:
    path = id2submission(problem_id, language, submission_id)

    if language == "C++":
        out = f"/tmp/{uuid.uuid4()}.out"
        try:
            subprocess.run(["g++", path, "-o", out], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            return None

        try:
            result = subprocess.run(
                [out],
                input=input_,
                timeout=timeout,
                capture_output=True,
                encoding="utf-8",
                errors="ignore",
                preexec_fn=lambda: resource.setrlimit(
                    resource.RLIMIT_AS,
                    (MAX_VIRTUAL_MEMORY, resource.RLIM_INFINITY),
                ),
                check=False,
            )

            stdout = str(result.stdout)

            if result.returncode == 0 and stdout != output:
                return "WA", "", stdout

            return str(result.returncode), str(result.stderr), stdout
        except subprocess.TimeoutExpired:
            return "TLE", "", ""

    if language == "Python":
        try:
            result = subprocess.run(
                ["python3", path],
                input=input_,
                timeout=timeout,
                capture_output=True,
                encoding="utf-8",
                errors="ignore",
                preexec_fn=lambda: resource.setrlimit(
                    resource.RLIMIT_AS,
                    (MAX_VIRTUAL_MEMORY, resource.RLIM_INFINITY),
                ),
                check=False,
            )

            stdout = str(result.stdout)

            if result.returncode == 0 and stdout != output:
                return "WA", "", stdout

            rs = "|".join(
                [
                    r"^(\w*Error:.*).*",
                    r"^(\w*Warning:.*).*",
                ]
            )
            p_class = re.compile(rs, re.MULTILINE)
            error_class = p_class.findall(result.stderr)
            if error_class:
                return (
                    functools.reduce(lambda acc, x: acc or x, error_class[0], None),
                    str(result.stderr),
                    stdout,
                )

            return str(result.returncode), str(result.stderr), stdout
        except subprocess.TimeoutExpired:
            return "TLE", "", ""

    raise NotImplementedError(f"{language} not implemented yet")


def codenet_download_data(force: bool = False) -> None:
    data_url = "https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0"
    tar_name = "Project_CodeNet.tar.gz"
    tar_path = os.path.join(INPUT_PATH, tar_name)

    if os.path.exists(ROOT_PATH) and not force:
        LOGGER.info("dataset root dir found at %s. skiping...", ROOT_PATH)
        return

    if not os.path.exists(tar_path) or force:
        LOGGER.info("download dataset from %s/%s", data_url, tar_name)
        wget.download(f"{data_url}/{tar_name}", out=tar_path)

    LOGGER.info("extract codenet to %s", INPUT_PATH)
    result = subprocess.run(
        f"tar -xzvf {tar_path} -C {INPUT_PATH}",
        capture_output=True,
        shell=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )

    LOGGER.info("process finished with code %s", result.returncode)


def codenet_filter_problems(force: bool = False) -> pd.DataFrame:
    """read the problem_list.csv file and remove the problems that
    do not have a sample input/output, then save the temporary csv file
    and return the problem_list as a dataframe.
    """
    out_file_path = os.path.join(GENERATED_PATH, "problem_list_clean.csv")

    if os.path.exists(out_file_path) and not force:
        LOGGER.info("dataset was already cleaned. skiping...")
        return pd.read_csv(out_file_path, index_col="id")

    file_path = os.path.join(META_PATH, "problem_list.csv")
    LOGGER.info("cleaning dataset from %s", file_path)

    problem_list_df = pd.read_csv(file_path, index_col="id")

    LOGGER.debug("fillna for time limit")
    problem_list_df["time_limit"].fillna(
        problem_list_df["time_limit"].median(), inplace=True
    )
    LOGGER.debug("fillna for memory limit")
    problem_list_df["memory_limit"].fillna(
        problem_list_df["memory_limit"].median(), inplace=True
    )

    LOGGER.debug("compute input mask")
    problem_ids = problem_list_df.index.unique()
    input_mask = [os.path.exists(id2inout(problem_id)) for problem_id in problem_ids]

    LOGGER.debug("remove problems that do not have input")
    problem_list_df = problem_list_df.loc[input_mask]
    problem_ids = problem_list_df.index.unique()

    LOGGER.debug("drop rating, targs and complexity")
    problem_list_df.drop(["rating", "tags", "complexity"], axis="columns", inplace=True)

    LOGGER.info("save cleaned dataset to %s", out_file_path)
    problem_list_df.to_csv(out_file_path)

    return problem_list_df


def codenet_submission_pairs_task(problem_id: str) -> pd.DataFrame:
    columns = [
        "problem_id",
        "language",
        "original_status",
        "pass",
        "fail",
        "change",
        "i1",
        "i2",
        "j1",
        "j2",
        "error",
        "stderr",
        "stdout",
    ]
    dfs = []

    with open(id2inout(problem_id, name="input"), "r", encoding="utf-8") as f:
        input_ = f.read()
    with open(id2inout(problem_id, name="output"), "r", encoding="utf-8") as f:
        output = f.read()

    problem_path = os.path.join(META_PATH, f"{problem_id}.csv")
    problem_df = pd.read_csv(problem_path, index_col="submission_id")
    if problem_df.empty:
        return pd.DataFrame()

    problem_df = problem_df[
        (problem_df["status"].isin(SUPPORTED_ERRORS))
        & (problem_df["language"].isin(SUPPORTED_LANGUAGES))
    ]

    grouped_languages = problem_df.groupby("language")
    for language, problem_df in grouped_languages:
        if problem_df.empty:
            continue

        submissions_diff_dfs = []
        grouped_users = problem_df.sort_values("date").groupby("user_id")
        for _, submission_df in grouped_users:
            if len(submission_df) < 2:
                continue

            submission_ids = submission_df.index.unique()
            for original_id, changed_id in zip(submission_ids, submission_ids[1:]):
                LOGGER.debug(
                    "Checking submission %s",
                    id2submission(problem_id, language, original_id),
                )

                if original_id in ["s237078692", "s717600459"]:
                    LOGGER.debug(
                        "Submission %s would crash the tool."
                        "You are so smart aren't you mister...",
                        id2submission(problem_id, language, original_id),
                    )
                    continue

                # Check if status is non-Accepted -> Accepted; otherwise skip
                original_status = submission_df.loc[original_id, "status"]
                changed_status = submission_df.loc[changed_id, "status"]
                if not (original_status != "Accepted" and changed_status == "Accepted"):
                    continue

                # Format the source code and read it; if there is an error skip
                original_ok, original_format_src = read_format_submission(
                    problem_id, language, original_id
                )
                changed_ok, changed_format_src = read_format_submission(
                    problem_id, language, changed_id
                )
                if not original_ok or not changed_ok:
                    continue

                # Check if there is only one line changed between the original
                # and the changed src; otherwise skip
                # diff = line_diff_checker(original_format_src, changed_format_src)
                # Check if there is only one chunk changed between the original
                # and the changed src; otherwise skip
                # diff = chunk_diff_checker(original_format_src, changed_format_src)
                # if diff is None:
                #     continue
                # Check if there is only one chunk of lines changed between the
                # original and the changed src; otherwise skip
                # diff = lines_diff_checker(original_format_src, changed_format_src)
                # if diff is None:
                #     continue
                # Check if there is only one chunk of lines changed between the
                # original and the changed src; if there are multiple chunks
                # merge everything from the start chunk to last chunk;
                diff = lines_diff_chunk_checker(original_format_src, changed_format_src)
                if diff is None:
                    continue
                change, i1, i2, j1, j2 = diff

                # Check with linter the original source code; if dumb code skip
                original_ok = linter_check_submission(problem_id, language, original_id)
                if not original_ok:
                    continue

                # Check and get the error annotations. If the original status
                # is MLE we will trust it :) because my pc crashes
                error_tup = None
                if original_status == "Memory Limit Exceeded":
                    error_tup = "MLE", "", ""
                elif original_status == "Time Limit Exceeded":
                    error_tup = "TLE", "", ""
                else:
                    error_tup = execute_source(
                        problem_id,
                        language,
                        original_id,
                        input_,
                        output,
                        timeout=2.0,
                    )
                if error_tup is None:
                    continue

                error, stderr, stdout = error_tup

                LOGGER.debug(
                    "Added %s to csv", id2submission(problem_id, language, original_id)
                )

                submissions_diff_dfs.append(
                    (
                        problem_id,
                        language,
                        original_status,
                        original_format_src,
                        changed_format_src,
                        change,
                        i1,
                        i2,
                        j1,
                        j2,
                        error,
                        stderr,
                        stdout,
                    )
                )

        df = pd.DataFrame(submissions_diff_dfs, columns=columns)
        dfs.append(df)

    return pd.DataFrame() if not dfs else pd.concat(dfs, ignore_index=True)


def codenet_submission_pairs(
    problem_list_df: pd.DataFrame, force: bool = False
) -> pd.DataFrame:
    generated_pairs_path = os.path.join(GENERATED_PATH, "generated_pairs.csv")

    if os.path.exists(generated_pairs_path) and not force:
        LOGGER.info("pairs already generated. skiping...")
        return pd.read_csv(generated_pairs_path, keep_default_na=False)

    tmp_paths = glob.glob(generated_pairs_path + ".tmp.*")
    problem_ids = problem_list_df.index.unique()
    count = problem_ids.shape[0]
    initial = 0

    dfs = []
    if tmp_paths:
        LOGGER.info("loading previous checkpoint...")
        xs = []
        for path in tmp_paths:
            try:
                xs.append(pd.read_csv(path))
            except Exception as exc:
                LOGGER.warning(
                    f"{path} generated an exception:"
                    + f"{exc}\ntraceback:\n{traceback.format_exc()}"
                )
        df = pd.concat(xs, ignore_index=True)
        check_problem_ids = [Path(path).suffix[1:] for path in tmp_paths]
        problem_ids = problem_ids[~problem_ids.isin(check_problem_ids)]
        initial = count - problem_ids.shape[0]
        dfs = [df]
    else:
        df = pd.DataFrame()

    with tqdm(total=count, initial=initial) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=P) as executor:
            future_to_problem_id = {
                executor.submit(codenet_submission_pairs_task, problem_id): problem_id
                for problem_id in problem_ids
            }

            for future in concurrent.futures.as_completed(future_to_problem_id):
                problem_id = future_to_problem_id[future]

                try:
                    problem_pairs_df = future.result()
                    dfs.append(problem_pairs_df)
                    problem_pairs_df.to_csv(
                        generated_pairs_path + f".tmp.{problem_id}",
                        escapechar="\\",
                        index=False,
                    )
                except Exception as exc:
                    LOGGER.error(
                        f"{problem_id} generated an exception:"
                        + f"{exc}\ntraceback:\n{traceback.format_exc()}"
                    )
                else:
                    pbar.set_description(f"[generate pairs] processing {problem_id}")
                    pbar.update(1)

    df = pd.concat(dfs, ignore_index=True).sort_values("problem_id")
    df.to_csv(generated_pairs_path, index=False)

    return df


def codenet_problem_statements(problem_list_df: pd.DataFrame, force: bool = False):
    problem_statements_path = os.path.join(GENERATED_PATH, "problem_descriptions")

    if os.path.exists(problem_statements_path) and not force:
        LOGGER.info("Problem Statements already filtered. skiping...")
        return

    os.makedirs(problem_statements_path, exist_ok=True)

    for problem_id in tqdm(problem_list_df.index):
        src = id2desc(problem_id)
        dst = os.path.join(problem_statements_path, os.path.basename(src))
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())


def codenet_prepare_kaggle(
    problem_list_df: pd.DataFrame, submission_pairs_df: pd.DataFrame
):
    LOGGER.info("creating kaggle zip...")

    kaggle_zip_path = os.path.join(GENERATED_PATH, "bugnet.zip")
    bugnet_descriptions_path = os.path.join(GENERATED_PATH, "problem_descriptions.json")
    bugnet_tests_path = os.path.join(GENERATED_PATH, "problem_tests.json")

    codex_pairs_df = submission_pairs_df.groupby("language").head(100)
    test_problem_ids = codex_pairs_df["problem_id"].unique()

    unique_problem_ids = [
        index for index in problem_list_df.index if index not in test_problem_ids
    ]
    unique_languages = submission_pairs_df["language"].unique()

    split_index = int(len(unique_problem_ids) * 0.8)

    train_problem_ids = unique_problem_ids[:split_index]
    valid_problem_ids = unique_problem_ids[split_index:]

    train_df = submission_pairs_df[
        submission_pairs_df["problem_id"].isin(train_problem_ids)
    ]
    valid_df = submission_pairs_df[
        submission_pairs_df["problem_id"].isin(valid_problem_ids)
    ]
    test_df = submission_pairs_df[
        submission_pairs_df["problem_id"].isin(test_problem_ids)
    ]

    descriptions = []
    for problem_id in tqdm(problem_list_df.index):
        src = id2desc(problem_id)
        with open(src, "r", encoding="utf-8") as desc:
            description = desc.read()
            descriptions.append({"problem_id": problem_id, "description": description})

    descriptions_df = pd.DataFrame(descriptions)
    descriptions_df.to_json(bugnet_descriptions_path, orient="records")

    test_cases = []
    for problem_id in tqdm(problem_list_df.index):
        with open(id2inout(problem_id, name="input"), "r", encoding="utf-8") as f:
            input_ = f.read()
        with open(id2inout(problem_id, name="output"), "r", encoding="utf-8") as f:
            output = f.read()

        test_cases.append({"problem_id": problem_id, "input": input_, "output": output})

    test_cases_df = pd.DataFrame(test_cases)
    test_cases_df.to_json(bugnet_tests_path, orient="records")

    with ZipFile(kaggle_zip_path, "w"):
        pass

    for language in unique_languages:
        lang_train_df = train_df[train_df["language"] == language]
        lang_valid_df = valid_df[valid_df["language"] == language]
        lang_test_df = test_df[test_df["language"] == language]

        bugnet_train_path = os.path.join(GENERATED_PATH, language + "_train.jsonl")
        bugnet_valid_path = os.path.join(GENERATED_PATH, language + "_validation.jsonl")
        bugnet_test_path = os.path.join(GENERATED_PATH, language + "_test.jsonl")

        lang_train_df.to_json(bugnet_train_path, orient="records", lines=True)
        lang_valid_df.to_json(bugnet_valid_path, orient="records", lines=True)
        lang_test_df.to_json(bugnet_test_path, orient="records", lines=True)

        with ZipFile(kaggle_zip_path, "a") as zip_obj:
            zip_obj.write(bugnet_train_path, os.path.basename(bugnet_train_path))
            zip_obj.write(bugnet_valid_path, os.path.basename(bugnet_valid_path))
            zip_obj.write(bugnet_test_path, os.path.basename(bugnet_test_path))

    with ZipFile(kaggle_zip_path, "a") as zip_obj:
        zip_obj.write(
            bugnet_descriptions_path,
            os.path.basename(bugnet_descriptions_path),
        )
        zip_obj.write(bugnet_tests_path, os.path.basename(bugnet_tests_path))


def main():
    codenet_download_data()
    problem_list_df = codenet_filter_problems()
    submission_pairs_df = codenet_submission_pairs(problem_list_df)
    codenet_problem_statements(problem_list_df)
    codenet_prepare_kaggle(problem_list_df, submission_pairs_df)


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
        "-P",
        "--cpu",
        default=P,
        dest="cpu",
        type=int,
        help="Number of CPU cores to use, default=cpu_count()",
    )
    args = parser.parse_args()

    formatter = logging.Formatter(
        fmt="%(levelname)s: %(asctime)s %(message)s",
        datefmt="%d/%m/%y %H:%M:%S",
    )
    s_handler = logging.StreamHandler()
    s_handler.setFormatter(formatter)
    LOGGER.addHandler(s_handler)

    f_handler = logging.FileHandler(os.getenv("LOG_FILE", LOG_PATH))
    f_handler.setFormatter(formatter)
    LOGGER.addHandler(f_handler)

    LOGGER.setLevel(levels[args.loglevel])

    os.makedirs(INPUT_PATH, exist_ok=True)
    os.makedirs(GENERATED_PATH, exist_ok=True)

    P = args.cpu

    main()
