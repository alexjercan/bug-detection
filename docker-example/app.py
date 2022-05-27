import streamlit as st

from inference import run, prepare_model
from inference import LIGHT_THEME, DARK_THEME


@st.cache(allow_output_mutation=True)
def load_model():
    return prepare_model()


def main():
    (
        tokenizer_ed,
        model_ed,
        tokenizer_tc,
        model_tc,
        tokenizer_cg,
        model_cg,
    ) = load_model()

    st.title("Buggy Source Code Detection")

    default_code = """A = map(input().split())
print(A[0])"""
    code = st.text_area("Enter the code here:", default_code)

    if st.button("Run"):
        results = run(
            code,
            tokenizer_ed,
            model_ed,
            tokenizer_tc,
            model_tc,
            tokenizer_cg,
            model_cg,
            theme=DARK_THEME,
        )
        results = [result.replace("\n</span>", "</span><br>") for result in results]
        st.session_state["index"] = 0
        st.session_state["results"] = results

    if (
        "results" in st.session_state
        and st.session_state["results"]
        and st.button("Next")
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
