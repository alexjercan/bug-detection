import os
import re
import io
import codecs
import tempfile

import pandas as pd

from util import handle_process

tools_path = "../Project_CodeNet/tools/"
tokenizer_dir_path = tools_path + "spt-generator/"
spt_profile = tokenizer_dir_path + "spt.profile"
tokenizer_path = tokenizer_dir_path + "scripts/run/tokenize.sh"

assert (
    "AI4CODE_HOME" in os.environ
), "You need to compile the AST Tokenizer and then source the spt.profile script\n"


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

    return pd.read_csv(io.StringIO(output), sep=",", keep_default_na=False)


def run_pythontokenizer(file_path: str) -> pd.DataFrame:
    cmd = f"{tokenizer_path} {file_path}"
    output, error, returncode = handle_process(cmd)
    assert returncode == 0, f"Error in tokenize {error} {output} {returncode} {cmd}"

    return pd.read_csv(io.StringIO(output), sep=",", keep_default_na=False)


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

    return pd.read_csv(io.StringIO(output), sep=",", keep_default_na=False)


def run_javatokenizer(file_path: str) -> pd.DataFrame:
    cmd = f"{tokenizer_path} {file_path}"
    output, error, returncode = handle_process(cmd)
    assert returncode == 0, f"Error in tokenize {error} {output} {returncode} {cmd}"

    return pd.read_csv(io.StringIO(output), sep=",", keep_default_na=False)


def run_tokenizer(file_path: str, language: str) -> pd.DataFrame:
    if language == "C":
        return run_ctokenizer(file_path)
    if language == "Python":
        return run_pythontokenizer(file_path)
    if language == "C++":
        return run_cpptokenizer(file_path)
    if language == "Java":
        return run_javatokenizer(file_path)

    assert False, "Error"


def apply_token_arrangements(
    original_tokens_df: pd.DataFrame,
    changed_tokens_df: pd.DataFrame,
    opcode: tuple[str, int, int, int, int],
) -> pd.DataFrame:
    tag, i1, i2, j1, j2 = opcode

    if tag == "insert":
        return changed_tokens_df.drop(range(j1, j2))

    first_df = changed_tokens_df[:j1]
    mid_df = original_tokens_df[i1:i2]
    second_df = changed_tokens_df[j2:]
    return pd.concat([first_df, mid_df, second_df]).reset_index(drop=True)


def generate_token_arrangements(
    original_tokens_df: pd.DataFrame,
    changed_tokens_df: pd.DataFrame,
    opcodes_df: pd.DataFrame,
) -> list[tuple[int, pd.DataFrame]]:
    return [
        (_id, apply_token_arrangements(original_tokens_df, changed_tokens_df, opcode))
        for _id, opcode in opcodes_df.iterrows()
    ]


def decode_escapes(s):
    ESCAPE_SEQUENCE_RE = re.compile(
        r"""
        ( \\U........      # 8-digit hex escapes
        | \\u....          # 4-digit hex escapes
        | \\x..            # 2-digit hex escapes
        | \\[0-7]{1,3}     # Octal escapes
        | \\N\{[^}]+\}     # Unicode characters by name
        | \\[\\'"abfnrtv]  # Single-character escapes
        )""",
        re.UNICODE | re.VERBOSE,
    )

    def decode_match(match):
        return codecs.decode(match.group(0), "unicode-escape")

    return ESCAPE_SEQUENCE_RE.sub(decode_match, s)


def tokens2str_python(tokens: list[str]) -> str:
    return "".join(tokens)


def tokens2str(df: pd.DataFrame, language: str) -> str:
    tokens = df["text"].values.tolist()[:-1]
    if language == "Python":
        return tokens2str_python(tokens)

    return None
