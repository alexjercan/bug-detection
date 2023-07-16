from transformers import (
    RobertaForTokenClassification,
    RobertaTokenizerFast,
    T5ForConditionalGeneration,
)


def prepare_model():
    _ = RobertaTokenizerFast.from_pretrained(
        "alexjercan/codet5-base-buggy-error-description"
    )
    _ = T5ForConditionalGeneration.from_pretrained(
        "alexjercan/codet5-base-buggy-error-description"
    )
    _ = RobertaTokenizerFast.from_pretrained(
        "alexjercan/codebert-base-buggy-token-classification"
    )
    _ = RobertaForTokenClassification.from_pretrained(
        "alexjercan/codebert-base-buggy-token-classification"
    )
    _ = RobertaTokenizerFast.from_pretrained("alexjercan/codet5-base-buggy-code-repair")
    _ = T5ForConditionalGeneration.from_pretrained(
        "alexjercan/codet5-base-buggy-code-repair"
    )


if __name__ == "__main__":
    prepare_model()
