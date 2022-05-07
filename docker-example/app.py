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
print(A)"""
    code = st.text_area("Enter the code here:", default_code)

    if st.button("Run"):
        result = run(
            code,
            tokenizer_ed,
            model_ed,
            tokenizer_tc,
            model_tc,
            tokenizer_cg,
            model_cg,
            theme=DARK_THEME,
        )[0]
        result = result.replace("\n</span>", "</span><br>")
        st.markdown(result, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
