import streamlit as st

from inference import run
from inference import LIGHT_THEME, DARK_THEME

from transformers import (
    RobertaTokenizerFast,
    T5ForConditionalGeneration,
    RobertaForTokenClassification,
)


@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer_ed = RobertaTokenizerFast.from_pretrained(
        "alexjercan/codet5-base-buggy-error-description"
    )
    model_ed = T5ForConditionalGeneration.from_pretrained(
        "alexjercan/codet5-base-buggy-error-description"
    )
    tokenizer_tc = RobertaTokenizerFast.from_pretrained(
        "alexjercan/codebert-base-buggy-token-classification"
    )
    model_tc = RobertaForTokenClassification.from_pretrained(
        "alexjercan/codebert-base-buggy-token-classification"
    )
    return tokenizer_ed, model_ed, tokenizer_tc, model_tc


def main():
    tokenizer_ed, model_ed, tokenizer_tc, model_tc = load_model()

    st.title("Buggy Source Code Detection")

    default_code = """A = map(input().split())
print(A)"""
    code = st.text_area("Enter the code here:", default_code)

    if st.button("Run"):
        result = run(
            code, tokenizer_ed, model_ed, tokenizer_tc, model_tc, theme=DARK_THEME
        )[0]
        result = result.replace("\n</span>", "</span><br>")
        st.markdown(result, unsafe_allow_html=True)


main()
