import re

import pandas as pd


def _is_punct(token):
    [token_class, token] = token
    return token_class == "punctuator" and token in [",", ";"]


def _is_literal(token):
    [token_class, _] = token
    return token_class in [
        "stringliteral",
        "integerconstant",
        "floatingconstant",
        "characterconstant",
    ]


def _is_keyword(token):
    [token_class, _] = token
    return token_class == "keyword"


def _is_binary(token):
    [token_class, token] = token
    return token_class == "punctuator" and token in [
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
    [token_class, token] = token
    return token_class == "punctuator" and token in ["~", "!", "[", "]"]


def _is_assign(token):
    [token_class, token] = token
    return token_class == "punctuator" and token in [
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
    return token_class == "identifier" and len(p.findall(token)) > 0


def _is_open_p(token):
    [token_class, token] = token
    return token_class == "punctuator" and token == "("


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


def mk_add_token_class(suffix="_x"):
    def add_token_class(group_df):
        token_class = get_token_class(group_df, suffix)
        group_df["token_class"] = token_class
        return group_df

    return add_token_class
