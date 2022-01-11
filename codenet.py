import os
import re
import io
import wget
import shutil
import tarfile
import tempfile
import functools
import traceback
import subprocess
import concurrent.futures

import pandas as pd

from tqdm import tqdm
from difflib import SequenceMatcher
from argparse import ArgumentParser

from myparser import mk_add_token_class
from mydifflib import group_diff_chunks, pdiff, single_change

from typing import Union

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

problem_list_clean_path = input_path + "generated/problem_list_clean.csv"
generated_pairs_path = input_path + "generated/generated_pairs.csv"
cleaned_generated_pairs_path = input_path + "generated/generated_pairs_tok.csv"
token_class_generated_pairs_path = (
    input_path + "generated/token_class_generated_pairs.csv"
)
clean_pairs_path = input_path + "generated/clean_pairs.csv"
error_pairs_path = input_path + "generated/error_pairs.csv"

supported_languages = ["C", "Python", "C++", "Java"]
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


def id2desc(problem_id: str) -> str:
    return descriptions_path + problem_id + ".html"


def id2inout(problem_id: str, name: str = "input") -> str:
    return derived_path + "input_output/data/" + problem_id + "/" + name + ".txt"


def id2submission(
    problem_id: str, language: str, submission_id: str, filename_ext: str
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
    problem_id: str, language: str, submission_id: str, extension: str
) -> list[str]:
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


def download_data() -> None:
    if os.path.exists(root_path):
        print("dataset root dir found")
        return

    if not os.path.exists(tar_path):
        wget.download(f"{data_url}/{tar_name}", out=tar_path)

    with tarfile.open(tar_path) as tf:
        tf.extractall(path=data_path)


def clean_problem_list(problem_list_df: pd.DataFrame = None) -> pd.DataFrame:
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

    input_mask = [
        os.path.exists(id2inout(str(problem_id))) for problem_id in problem_ids
    ]

    problem_list_df = problem_list_df.loc[input_mask]
    problem_ids = problem_list_df.index.unique()

    return problem_list_df


def preprocess_problem_for_language(
    problem_df: pd.DataFrame, problem_id: str, language: str = "C", extension: str = "c"
) -> list[tuple[str, str, int, str, int, str, str]]:
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

            original_language = submission_df.loc[original_id, "original_language"]
            submissions_diff_dfs.append(
                (
                    original_id,
                    changed_id,
                    chunks[0][0],
                    chunks[0][2],
                    chunks[0][3],
                    original_status,
                    original_language,
                )
            )

    return submissions_diff_dfs


def generate_pairs_task(problem_id: str) -> pd.DataFrame:
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


def generate_pairs(problem_list_df: pd.DataFrame = None) -> pd.DataFrame:
    if problem_list_df is None:
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

    return df.sort_values("original_id")


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


def run_ctokenizer(file_path: str) -> pd.DataFrame:
    grep_command = (
        "grep -P -v '^[ \\t]*#[ \\t]*(include|import)[ \\t]*[\\<|\\\"].*(?<!\\*\\/)$'"
    )

    with tempfile.NamedTemporaryFile("w+t", suffix=".c") as writer:
        cmd = f"{grep_command} {file_path} | gcc -E -P -xc++ - -o {writer.name}"
        output, error, returncode = handle_process(cmd)
        assert (
            returncode == 0
        ), f"Error in grep and gcc {error} {output} {returncode} {cmd}"
        output, error, returncode = handle_process(f"{tokenizer_path} {writer.name}")
        assert returncode == 0, f"Error in tokenize {error} {output} {returncode}"

    return pd.read_csv(io.StringIO(output), sep=",")


def run_pythontokenizer(file_path: str) -> pd.DataFrame:
    cmd = f"{tokenizer_path} {file_path}"
    output, error, returncode = handle_process(cmd)
    assert returncode == 0, f"Error in tokenize {error} {output} {returncode} {cmd}"

    return pd.read_csv(io.StringIO(output), sep=",")


def run_cpptokenizer(file_path: str) -> pd.DataFrame:
    grep_command = (
        "grep -P -v '^[ \\t]*#[ \\t]*(include|import)[ \\t]*[\\<|\\\"].*(?<!\\*\\/)$'"
    )

    with tempfile.NamedTemporaryFile("w+t", suffix=".cpp") as writer:
        cmd = f"{grep_command} {file_path} | gcc -E -P -xc - -o {writer.name}"
        output, error, returncode = handle_process(cmd)
        assert (
            returncode == 0
        ), f"Error in grep and gcc {error} {output} {returncode} {cmd}"
        output, error, returncode = handle_process(f"{tokenizer_path} {writer.name}")
        assert returncode == 0, f"Error in tokenize {error} {output} {returncode}"

    return pd.read_csv(io.StringIO(output), sep=",")


def run_javatokenizer(file_path: str) -> pd.DataFrame:
    cmd = f"{tokenizer_path} {file_path}"
    output, error, returncode = handle_process(cmd)
    assert returncode == 0, f"Error in tokenize {error} {output} {returncode} {cmd}"

    return pd.read_csv(io.StringIO(output), sep=",")


def run_tokenizer(
    problem_id: str, language: str, submission_id: str, filename_ext: str
) -> pd.DataFrame:
    file_path = id2submission(problem_id, language, submission_id, filename_ext)

    if language == "C":
        return run_ctokenizer(file_path)
    if language == "Python":
        return run_pythontokenizer(file_path)
    if language == "C++":
        return run_cpptokenizer(file_path)
    if language == "Java":
        return run_javatokenizer(file_path)

    assert False, "Error"


def tokenize_generated_pairs_task(
    original_id: str,
    changed_id: str,
    original_line: int,
    diff_op: str,
    changed_line: int,
    original_status: str,
    original_language: str,
    problem_id: str,
    language: str,
    filename_ext: str,
) -> pd.DataFrame:
    original_tokens_df = run_tokenizer(problem_id, language, original_id, filename_ext)
    changed_tokens_df = run_tokenizer(problem_id, language, changed_id, filename_ext)

    a = original_tokens_df["text"].values.tolist()
    b = changed_tokens_df["text"].values.tolist()
    s: SequenceMatcher = SequenceMatcher(None, a, b)
    opcodes = [x for x in s.get_opcodes() if x[0] != "equal"]
    if len(opcodes) != 1:
        return pd.DataFrame()

    tag, i1, i2, j1, j2 = opcodes[0]
    df = original_tokens_df.loc[i1:i2].merge(
        changed_tokens_df.loc[j1:j2], how="outer", on=["seqnr"]
    )
    df["tag"] = tag
    df["problem_id"] = problem_id
    df["original_id"] = original_id
    df["changed_id"] = changed_id
    df["language"] = language
    df["extension"] = filename_ext
    df["original_language"] = original_language
    df["original_status"] = original_status

    return df


def tokenize_generated_pairs(generated_pairs_df: pd.DataFrame = None) -> pd.DataFrame:
    if generated_pairs_df is None:
        generated_pairs_df = pd.read_csv(generated_pairs_path)
    submissions_diff_dfs = []

    with tqdm(total=len(generated_pairs_df)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=P) as executor:
            future_to_problem_id = {
                executor.submit(tokenize_generated_pairs_task, *row): row
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


def add_token_class(generated_pairs_df: pd.DataFrame = None) -> pd.DataFrame:
    if generated_pairs_df is None:
        generated_pairs_df = pd.read_csv(cleaned_generated_pairs_path)

    token_pairs_df = generated_pairs_df.groupby(
        ["original_id", "changed_id"]
    ).progress_apply(mk_add_token_class())

    return token_pairs_df


def exec_c(
    file_path: str, input: str = None, timeout: float = 2.0
) -> tuple[str, str, int]:
    with tempfile.NamedTemporaryFile("w+b", suffix=".out", delete=False) as writer:
        output, error, returncode = handle_process(
            f"gcc {file_path} -lm -w -O3 -o {writer.name}"
        )
        assert returncode == 0, f"Error in gcc {error} {output} {returncode}"

    result = handle_process([writer.name], input, timeout)
    os.unlink(writer.name)

    return result


def exec_python(
    file_path: str, input: str = None, timeout: float = 2.0
) -> tuple[str, str, int]:
    return handle_process(["python3", file_path], input, timeout)


def exec_cpp(
    file_path: str, input: str = None, timeout: float = 2.0
) -> tuple[str, str, int]:
    with tempfile.NamedTemporaryFile("w+b", suffix=".out", delete=False) as writer:
        output, error, returncode = handle_process(
            f"g++ {file_path} -lm -w -O3 -o {writer.name}"
        )
        assert returncode == 0, f"Error in g++ {error} {output} {returncode}"

    result = handle_process([writer.name], input, timeout)
    os.unlink(writer.name)

    return result


def exec_java(
    file_path: str, input: str = None, timeout: float = 2.0
) -> tuple[str, str, int]:
    with tempfile.TemporaryDirectory() as dir_path:
        file_path = shutil.copy(file_path, dir_path + "/Main.java")
        output, error, returncode = handle_process(f"javac -d {dir_path} {file_path}")
        assert returncode == 0, f"Error in javac {error} {output} {returncode}"

        classname = os.listdir(dir_path)[0].split(".")[0]
        return handle_process(f"java -classpath {dir_path} {classname}")


def exec_file(
    file_path: str, input: str = None, timeout: float = 2.0, language: str = None
) -> tuple[str, str, int]:
    if language == "C":
        return exec_c(file_path, input, timeout)
    if language == "Python":
        return exec_python(file_path, input, timeout)
    if language == "C++":
        return exec_cpp(file_path, input, timeout)
    if language == "Java":
        return exec_java(file_path, input, timeout)
    raise NotImplementedError


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
    rs = r"Exception in thread \".*\" (.*)"

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


def add_error_description_task(
    _id: int,
    time_limit: float,
    original_id: str,
    changed_id: str,
    original_line: int,
    diff_op: str,
    changed_line: int,
    original_status: str,
    original_language: str,
    problem_id: str,
    language: str,
    filename_ext: str,
) -> tuple[int, str, str, int]:
    file_path = id2submission(problem_id, language, original_id, filename_ext)

    input_path = id2inout(problem_id, name="input")
    output_path = id2inout(problem_id, name="output")

    with open(input_path, "r") as f:
        input = f.read()

    timeout = time_limit / 1000 * 2

    try:
        output, error, returncode = exec_file(file_path, input, timeout, language)
    except AssertionError as exc:
        output = ""
        returncode = 1
        error = f"Error: {exc}"

    error_class = extract_error_class((language, error, returncode))
    error_class_extra = extract_error_class_extra((language, error, returncode))

    return (_id, output, error, returncode, error_class, error_class_extra)


def add_error_description(
    clean_pairs_df: pd.DataFrame = None, problem_list_df: pd.DataFrame = None
) -> pd.DataFrame:
    if clean_pairs_df is None:
        clean_pairs_df = pd.read_csv(clean_pairs_path)
    if problem_list_df is None:
        problem_list_df = pd.read_csv(problem_list_clean_path, index_col="id")

    errs = []

    time_limit_f = lambda pid: problem_list_df.loc[pid]["time_limit"]

    with tqdm(total=len(clean_pairs_df)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=P) as executor:
            future_to_problem_id = {
                executor.submit(
                    add_error_description_task,
                    _id,
                    time_limit_f(row["problem_id"]),
                    *row,
                ): [_id, time_limit_f(row["problem_id"]), *row]
                for _id, row in clean_pairs_df.iterrows()
            }

            for future in concurrent.futures.as_completed(future_to_problem_id):
                (
                    _id,
                    time_limit,
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
                    err = future.result()
                    errs.append(err)
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

    return clean_pairs_df.join(errs_df)


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
        "--tokenizepairs",
        help="tokenize the generated source code pairs",
        action="store_true",
    )
    parser.add_argument(
        "--tokenclass",
        help="add the token class for the generated source code pairs",
        action="store_true",
    )
    parser.add_argument(
        "--errorclass",
        help="add the error class for the generated source code pairs",
        action="store_true",
    )
    parser.add_argument("-P", help="number of processors to use", default=4, type=int)

    args = parser.parse_args()

    assert (
        "AI4CODE_HOME" in os.environ
    ), "You need to compile the AST Tokenizer and then source the spt.profile script\n"

    P = args.P

    if args.download:
        download_data()
    if args.cleanlist:
        clean_problem_list().to_csv(problem_list_clean_path)
    if args.genpairs:
        generate_pairs().to_csv(generated_pairs_path, index=False)
    if args.tokenizepairs:
        tokenize_generated_pairs().to_csv(cleaned_generated_pairs_path, index=False)
    if args.tokenclass:
        add_token_class().to_csv(token_class_generated_pairs_path, index=False)
    if args.errorclass:
        add_error_description().to_csv(error_pairs_path, index=False)
