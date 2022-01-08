import re
from typing import Callable

import pandas as pd


def _is_punct(token):
    [_, token] = token
    return token in [",", ";"]


def _is_literal(token):
    [token_class, _] = token
    return token_class in [
        "stringliteral",
        "integerconstant",
        "floatingconstant",
        "characterconstant",
        "floating",
        "string",
        "character",
        "integer",
    ]


def _is_keyword(token):
    [token_class, _] = token
    return token_class in ["keyword", "keywords"]


def _is_binary(token):
    [_, token] = token
    return token in [
        "+",
        "*",
        "-",
        "/",
        "%",
        "<<",
        ">>",
        "^",
        "&",
        "|",
        "&&",
        "||",
        ">=",
        ">",
        "==",
        "<=",
        "<",
    ]


def _is_unary(token):
    [_, token] = token
    return token in ["~", "!", "[", "]"]


def _is_assign(token):
    [_, token] = token
    return token in [
        "=",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "<<=",
        ">>=",
        "^=",
        "&=",
        "|=",
        "++",
        "--",
        "!=",
    ]


def _is_identifier(token):
    [token_class, token] = token
    p = re.compile("^[A-Za-z_]\w*")
    return token_class in ["identifier", "name"] and len(p.findall(token)) > 0


def _is_open_p(token):
    [_, token] = token
    return token == "("


def check_token(func, tokens, n):
    if len(tokens) <= n:
        return False
    return func(tokens[n])


def check_tokens(func, tokens):
    return any([func(token) for token in tokens])


def check_sequence(funcs, tokens):
    for i in range(len(tokens) - len(funcs) + 1):
        if all([func(token) for (func, token) in zip(funcs, tokens[i:])]):
            return True
    return False


def negate(func):
    def f(token):
        return not func(token)

    return f


_function_call_seq = [_is_identifier, _is_open_p]
_variable_assign_seq = [_is_identifier, _is_assign]


def get_token_class(token_df, suffix="_x"):
    tokens = token_df[["class" + suffix, "text" + suffix]].dropna().values.tolist()
    if not tokens:
        return None

    if check_sequence(_function_call_seq, tokens):
        return "call"
    if check_sequence(_variable_assign_seq, tokens) or check_tokens(_is_assign, tokens):
        return "assign"
    if check_tokens(_is_binary, tokens):
        return "binary"
    if check_tokens(_is_unary, tokens):
        return "unary"
    if check_tokens(_is_literal, tokens):
        return "literal"
    if check_tokens(_is_identifier, tokens):
        return "identifier"
    if check_tokens(_is_keyword, tokens):
        return "keyword"
    if check_tokens(_is_punct, tokens):
        return "punctuator"

    return None


def mk_add_token_class(suffix: str = "_x") -> Callable[[pd.DataFrame], pd.DataFrame]:
    def add_token_class(group_df: pd.DataFrame) -> pd.DataFrame:
        token_class = get_token_class(group_df, suffix)
        group_df["token_class"] = token_class
        return group_df

    return add_token_class


def get_id_from_df(token_df):
    tokens = token_df[["class_x", "text_x"]].dropna().values.tolist()
    if not tokens:
        return None

    for i in range(len(token_df)):
        if check_token(_is_identifier, tokens, i):
            return tokens[i][1]
    return None


def get_fcall_from_df(token_df):
    tokens = token_df[["class_x", "text_x"]].dropna().values.tolist()
    if not tokens:
        return None

    for i in range(len(token_df)):
        if check_token(_is_identifier, tokens, i) and check_token(
            _is_open_p, tokens, i + 1
        ):
            return tokens[i][1]
    return None


def get_bop_from_df(token_df):
    tokens = token_df[["class_x", "text_x"]].dropna().values.tolist()
    if not tokens:
        return None

    for i in range(len(token_df)):
        if check_token(_is_binary, tokens, i):
            return tokens[i][1]
    return None


def get_assign_from_df(token_df):
    tokens = token_df[["class_x", "text_x"]].dropna().values.tolist()
    if not tokens:
        return None

    for i in range(len(token_df)):
        if check_token(_is_identifier, tokens, i) and check_token(
            _is_assign, tokens, i + 1
        ):
            return tokens[i][1]
    return None

def get_keyword_from_df(token_df):
    tokens = token_df[["class_x", "text_x"]].dropna().values.tolist()
    if not tokens:
        return None

    for i in range(len(token_df)):
        if check_token(_is_keyword, tokens, i):
            return tokens[i][1]
    return None


def get_unary_from_df(token_df):
    tokens = token_df[["class_x", "text_x"]].dropna().values.tolist()
    if not tokens:
        return None

    for i in range(len(token_df)):
        if check_token(_is_unary, tokens, i):
            return tokens[i][1]
    return None


def get_punct_from_df(token_df):
    tokens = token_df[["class_x", "text_x"]].dropna().values.tolist()
    if not tokens:
        return None

    for i in range(len(token_df)):
        if check_token(_is_punct, tokens, i):
            return tokens[i][1]
    return None
