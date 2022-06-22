import streamlit as st
import numpy as np

from inference import Session
from difflib import SequenceMatcher
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
        text += f'<span style="color:{accent_color if mask[i] == 1 else color};">{char}</span>'
    return "<pre>" + text + "</pre>"


def view(
    source_code: str,
    error: List[str],
    labels: List[List[int]],
    new_source_code: List[str],
    theme=LIGHT_THEME,
    beam_size_ed=5,
    beam_size_cg=1,
):
    source_code = [source_code for _ in range(beam_size_ed)]
    source_code_html = [
        color_source(src, labels[i], **theme)
        for i, src in enumerate(source_code)
        for _ in range(beam_size_ed)
    ]
    error_html = [f"<pre>{err}</pre>" for err in error for _ in range(beam_size_cg)]

    source_code = [src for src in source_code for _ in range(beam_size_cg)]
    new_source_code_html = [
        color_source(
            new_src, generate_char_mask(new_src, src), accent_color="green", **theme
        )
        for new_src, src in zip(new_source_code, source_code)
    ]

    results = []
    for src, err, new_src in zip(source_code_html, error_html, new_source_code_html):
        results.append(
            f"<h1>Source code</h1>{src}<h1>Error description</h1>{err}<h1>Repaired code</h1>{new_src}"
        )

    results = [result.replace("\n</span>", "</span><br>") for result in results]
    return results


@st.cache(allow_output_mutation=True)
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
            [src for srcs in new_source_code[0] for src in srcs],
            theme=THEME,
            beam_size_ed=5,
            beam_size_cg=1,
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
