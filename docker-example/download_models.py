from transformers import (
    RobertaTokenizerFast,
    T5ForConditionalGeneration,
    RobertaForTokenClassification,
)

def prepare_model():
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
    tokenizer_cg = RobertaTokenizerFast.from_pretrained(
        "alexjercan/codet5-base-masked-buggy-code-repair"
    )
    model_cg = T5ForConditionalGeneration.from_pretrained(
        "alexjercan/codet5-base-masked-buggy-code-repair"
    )


if __name__ == "__main__":
    prepare_model()
