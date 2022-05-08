import numpy as np

from typing import Union, List
from transformers import (
    RobertaTokenizerFast,
    T5ForConditionalGeneration,
    RobertaForTokenClassification,
)


def predict_error_description(tokenizer: RobertaTokenizerFast, model: T5ForConditionalGeneration, source: List[str]):
    tokenized_inputs = tokenizer(
        source,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    tokenized_labels = model.generate(**tokenized_inputs).cpu().detach().numpy()

    return tokenizer.batch_decode(tokenized_labels, skip_special_tokens=True)


def predict_token_class(tokenizer: RobertaTokenizerFast, model: RobertaForTokenClassification, error: List[str], source: List[str]):
    if isinstance(source, str):
        source = [source]
    if isinstance(error, str):
        error = [error]

    tokenized_inputs = tokenizer(
        text=error,
        text_pair=source,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    tokenized_labels = np.argmax(
        model(**tokenized_inputs)["logits"].cpu().detach().numpy(), 2
    )

    all_labels = []
    for i in range(tokenized_labels.shape[0]):
        labels = [0] * len(source[i])
        for j, label in enumerate(tokenized_labels[i]):
            if tokenized_inputs.token_to_sequence(i, j) != 1:
                continue

            word_id = tokenized_inputs.token_to_word(i, j)
            cs = tokenized_inputs.word_to_chars(i, word_id, sequence_index=1)
            if cs.start == cs.end:
                continue
            labels[cs.start : cs.end] |= tokenized_labels[i, j]

        all_labels.append([int(l) for l in labels])

    return all_labels


def predict_source_code(tokenizer: RobertaTokenizerFast, model: T5ForConditionalGeneration, error: List[str], source: List[str]):
    tokenized_inputs = tokenizer(
        text=error,
        text_pair=source,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    tokenized_labels = (
        model.generate(max_length=512, **tokenized_inputs).cpu().detach().numpy()
    )

    return tokenizer.batch_decode(tokenized_labels, skip_special_tokens=True)


class Session:
    def __init__(self):
        self.tokenizer_ed = RobertaTokenizerFast.from_pretrained(
            "alexjercan/codet5-base-buggy-error-description"
        )
        self.model_ed = T5ForConditionalGeneration.from_pretrained(
            "alexjercan/codet5-base-buggy-error-description"
        )
        self.tokenizer_tc = RobertaTokenizerFast.from_pretrained(
            "alexjercan/codebert-base-buggy-token-classification"
        )
        self.model_tc = RobertaForTokenClassification.from_pretrained(
            "alexjercan/codebert-base-buggy-token-classification"
        )
        self.tokenizer_cg = RobertaTokenizerFast.from_pretrained(
            "alexjercan/codet5-base-buggy-code-repair"
        )
        self.model_cg = T5ForConditionalGeneration.from_pretrained(
            "alexjercan/codet5-base-buggy-code-repair"
        )

    def run(self, source_code: Union[str, List[str]]):
        if isinstance(source_code, str):
            source_code = [source_code]

        print("Predicting error description...")
        error_description = predict_error_description(
            self.tokenizer_ed, self.model_ed, source_code
        )
        print("Predicting token class...")
        token_class = predict_token_class(
            self.tokenizer_tc, self.model_tc, error_description, source_code
        )
        print("Predicting source code...")
        source_code = predict_source_code(
            self.tokenizer_cg, self.model_cg, error_description, source_code
        )

        return error_description, token_class, source_code


if __name__ == "__main__":
    session = Session()
    print(session.run("A = map(input().split())\nprint(A)"))
