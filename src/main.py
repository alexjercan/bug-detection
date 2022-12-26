import logging
import os

import concurrent.futures
import pandas as pd
import subprocess
import traceback
import wget
from difflib import SequenceMatcher
from multiprocessing import cpu_count
from tqdm.auto import tqdm
from typing import Optional, Tuple

P = int(os.environ.get("CODENET_CPUS", cpu_count()))

SUPPORTED_LANGUAGES = ["C++"]
EXTENSIONS = {"C++": "cpp"}

SUPPORTED_ERRORS = ["Accepted", "Runtime Error", "Time Limit Exceeded", "Memory Limit Exceeded"]

assert set(SUPPORTED_LANGUAGES).issubset(set(EXTENSIONS)), (
    "Expected to have extension for all supported languages, "
    "but the following languages are missing: "
    f"{', '.join(set(SUPPORTED_LANGUAGES).difference(set(EXTENSIONS)))}"
)

INPUT_PATH = "input"
ROOT_PATH = os.path.join(INPUT_PATH, "Project_CodeNet")
DATA_PATH = os.path.join(ROOT_PATH, "data")
META_PATH = os.path.join(ROOT_PATH, "metadata")
DERIVED_PATH = os.path.join(ROOT_PATH, "derived")
GENERATED_PATH = os.path.join(INPUT_PATH, "generated")


def id2inout(problem_id: str, name: str = "input") -> str:
    return os.path.join(DERIVED_PATH, "input_output", "data", problem_id, f"{name}.txt")


def id2submission(
    problem_id: str,
    language: str,
    submission_id: str,
) -> str:
    return os.path.join(
        DATA_PATH, problem_id, language, f"{submission_id}.{EXTENSIONS[language]}"
    )


def read_format_submission(
    problem_id: str, language: str, submission_id: str
) -> Tuple[bool, str]:
    path = id2submission(problem_id, language, submission_id)

    if language == "C++":
        result = subprocess.run(
            ["clang-format", path],
            capture_output=True,
            encoding="utf-8",
            errors="ignore",
        )
        return result.returncode == 0, result.stdout

    raise NotImplementedError(f"{language} not implemented yet")


def line_diff_checker(original_src: str, changed_src: str) -> Optional[Tuple[str, int]]:
    original_src_lines = original_src.splitlines()
    changed_src_lines = changed_src.splitlines()

    opcodes = SequenceMatcher(None, original_src_lines, changed_src_lines).get_opcodes()
    changes = list(filter(lambda opcode: opcode[0] != "equal", opcodes))
    if len(changes) != 1:
        return None

    change, i1, i2, j1, j2 = changes[0]
    if i2 - i1 > 1 or j2 - j1 > 1:
        return None

    return change, i1


def linter_check_submission(problem_id: str, language: str, submission_id: str) -> bool:
    path = id2submission(problem_id, language, submission_id)

    if language == "C++":
        result = subprocess.run(
            ["clang-tidy", path],
            capture_output=True,
            encoding="utf-8",
            errors="ignore",
        )
        return result.returncode == 0

    raise NotImplementedError(f"{language} not implemented yet")


def codenet_download_data(force: bool = False) -> None:
    data_url = "https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0"
    tar_name = "Project_CodeNet.tar.gz"
    tar_path = os.path.join(INPUT_PATH, tar_name)

    if os.path.exists(ROOT_PATH) and not force:
        logging.info(f"dataset root dir found at {ROOT_PATH}. skiping...")
        return

    if not os.path.exists(tar_path) or force:
        logging.debug(f"download dataset from {data_url}/{tar_name}")
        wget.download(f"{data_url}/{tar_name}", out=tar_path)

    logging.debug(f"extract codenet to {INPUT_PATH}")
    result = subprocess.run(
        f"tar -xzvf {tar_path} -C {INPUT_PATH}",
        capture_output=True,
        shell=True,
        encoding="utf-8",
        errors="ignore",
    )

    logging.debug(f"process finished with status {result.returncode}")


def codenet_filter_problems(force: bool = False) -> pd.DataFrame:
    """read the problem_list.csv file and remove the problems that
    do not have a sample input/output, then save the temporary csv file
    and return the problem_list as a dataframe.
    """
    out_file_path = os.path.join(GENERATED_PATH, "problem_list_clean.csv")

    if os.path.exists(out_file_path) and not force:
        logging.info("dataset was already cleaned. skiping...")
        return pd.read_csv(out_file_path, index_col="id")

    file_path = os.path.join(META_PATH, "problem_list.csv")
    logging.debug(f"cleaning {file_path}")

    problem_list_df = pd.read_csv(file_path, index_col="id")

    logging.debug("fillna for time limit")
    problem_list_df["time_limit"].fillna(
        problem_list_df["time_limit"].median(), inplace=True
    )
    logging.debug("fillna for memory limit")
    problem_list_df["memory_limit"].fillna(
        problem_list_df["memory_limit"].median(), inplace=True
    )

    logging.debug("compute input mask")
    problem_ids = problem_list_df.index.unique()
    input_mask = [os.path.exists(id2inout(problem_id)) for problem_id in problem_ids]

    logging.debug("remove problems that do not have input")
    problem_list_df = problem_list_df.loc[input_mask]
    problem_ids = problem_list_df.index.unique()

    logging.debug("drop rating, targs and complexity")
    problem_list_df.drop(["rating", "tags", "complexity"], axis="columns", inplace=True)

    logging.debug(f"save to {out_file_path}")
    problem_list_df.to_csv(out_file_path)

    return problem_list_df


def codenet_submission_pairs_task(problem_id: str) -> pd.DataFrame:
    columns = [
        "problem_id",
        "language",
        "original_status",
        "original_src",
        "changed_src",
        "change",
        "line",
    ]
    dfs = []

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
        for user_id, submission_df in grouped_users:
            if len(submission_df) < 2:
                continue

            submission_ids = submission_df.index.unique()
            for original_id, changed_id in zip(submission_ids, submission_ids[1:]):
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
                diff = line_diff_checker(original_format_src, changed_format_src)
                if diff is None:
                    continue
                change, line = diff

                # Check with linter the original source code; if dumb code skip
                original_ok = linter_check_submission(problem_id, language, original_id)
                if not original_ok:
                    continue

                submissions_diff_dfs.append(
                    (
                        problem_id,
                        language,
                        original_status,
                        original_format_src,
                        changed_format_src,
                        change,
                        line,
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
        logging.info("pairs already generated. skiping...")
        return pd.read_csv(generated_pairs_path, keep_default_na=False)

    problem_pairs = []
    problem_ids = problem_list_df.index.unique()
    with tqdm(total=len(problem_ids)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=P) as executor:
            future_to_problem_id = {
                executor.submit(codenet_submission_pairs_task, problem_id): problem_id
                for problem_id in problem_ids
            }

            for future in concurrent.futures.as_completed(future_to_problem_id):
                problem_id = future_to_problem_id[future]

                try:
                    problem_pairs_df = future.result()
                    problem_pairs.append(problem_pairs_df)
                except Exception as exc:
                    logging.error(
                        f"{problem_id} generated an exception:"
                        + f"{exc}\ntraceback:\n{traceback.format_exc()}"
                    )
                else:
                    pbar.set_description(f"[generate pairs] processing {problem_id}")
                    pbar.update(1)

    df = pd.concat(problem_pairs, ignore_index=True).sort_values("problem_id")
    df.to_csv(generated_pairs_path, index=False)

    return df


if __name__ == "__main__":
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.getenv("LOG_FILE", "codenet.log")),
        ],
        level=logging.DEBUG,
        format="%(levelname)s: %(asctime)s %(message)s",
        datefmt="%d/%m/%y %H:%M:%S",
    )

    os.makedirs(INPUT_PATH, exist_ok=True)
    os.makedirs(GENERATED_PATH, exist_ok=True)

    codenet_download_data()
    problem_list_df = codenet_filter_problems()
    submission_pairs_df = codenet_submission_pairs(problem_list_df)
