import numpy as np
import streamlit as st
from difflib import SequenceMatcher
from inference import Session
from typing import List

LIGHT_THEME = {"norm_color": "black", "ws_color": "lightgrey"}
DARK_THEME = {"norm_color": "white", "ws_color": "grey"}
THEME = DARK_THEME


def generate_char_mask(original_src: str, changed_src: str) -> List[int]:
    s = SequenceMatcher(None, original_src, changed_src)
    opcodes = [x for x in s.get_opcodes() if x[0] != "equal"]

    original_labels = np.zeros_like(list(original_src), dtype=np.int32)
    for _, i1, i2, _, _ in opcodes:
        original_labels[i1 : max(i1 + 1, i2)] = 1

    return original_labels.tolist()


def color_source(
    source_code: str,
    mask: List[int],
    accent_color="red",
    norm_color="black",
    ws_color="lightgrey",
) -> str:
    text = ""
    for i, char in enumerate(source_code):
        color = norm_color
        if char == " ":
            char = "•"
            color = ws_color
        if char == "\n":
            char = "↵\n"
            color = ws_color

        col = accent_color if mask[i] == 1 else color
        text += f'<span style="color:{col};">{char}</span>'
    return "<pre>" + text + "</pre>"


def view(
    source_code: str,
    error: List[str],
    labels: List[List[int]],
    new_source_code: List[List[str]],
    theme=None,
    beam_size_ed=5,
):
    if theme is None:
        theme = LIGHT_THEME

    source_codes = [source_code for _ in range(beam_size_ed)]
    source_code_html = [
        color_source(src, labels[i], **theme) for i, src in enumerate(source_codes)
    ]
    error_html = [f"<pre>{err}</pre>" for err in error]

    new_source_code_html = []
    for i, new_srcs in enumerate(new_source_code):
        new_source_code_html.append(
            [
                color_source(
                    new_src,
                    generate_char_mask(new_src, source_codes[i]),
                    accent_color="green",
                    **theme,
                )
                for new_src in new_srcs
            ]
        )

    results = []
    for i, new_srcs in enumerate(new_source_code_html):
        results.extend(
            [
                f"<h1>Source code</h1>{source_code_html[i]}"
                f"<h1>Error description</h1>{error_html[i]}"
                f"<h1>Repaired code</h1>{new_src}"
                for new_src in new_srcs
            ]
        )
    results = [result.replace("\n</span>", "</span><br>") for result in results]
    return results


@st.cache_resource
def load_model():
    return Session()


def main():
    session = load_model()

    st.title("Buggy Source Code Detection")

    default_code = """A = map(input().split())
print(A[0])"""
    code = st.text_area("Enter the code here:", default_code)

    col1, col2, col3 = st.columns(3)

    if col1.button("Run"):
        errors, labels, new_source_code = session.run(
            [code], beam_size_ed=5, beam_size_cg=1
        )
        results = view(
            code,
            errors[0],
            labels[0],
            new_source_code[0],
            theme=THEME,
            beam_size_ed=5,
        )
        st.session_state["index"] = 0
        st.session_state["results"] = results

    if (
        "results" in st.session_state
        and st.session_state["results"]
        and col2.button("Back")
    ):
        index = st.session_state["index"]
        results = st.session_state["results"]

        index = (index - 1) % len(results)
        st.session_state["index"] = index

    if (
        "results" in st.session_state
        and st.session_state["results"]
        and col3.button("Next")
    ):
        index = st.session_state["index"]
        results = st.session_state["results"]

        index = (index + 1) % len(results)
        st.session_state["index"] = index

    if "results" in st.session_state and st.session_state["results"]:
        index = st.session_state["index"]
        results = st.session_state["results"]

        st.markdown(results[index], unsafe_allow_html=True)


if __name__ == "__main__":
    main()
