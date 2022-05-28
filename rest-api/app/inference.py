import numpy as np

from typing import Union, List
from transformers import (
    RobertaTokenizerFast,
    T5ForConditionalGeneration,
    RobertaForTokenClassification,
)

BEAM_SIZE = 5


def predict_error_description(tokenizer: RobertaTokenizerFast, model: T5ForConditionalGeneration, source: List[str], beam_size=BEAM_SIZE) -> List[List[str]]:
    tokenized_inputs = tokenizer(
        source,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    tokenized_labels = (
        model.generate(
            num_beams=beam_size,
            no_repeat_ngram_size=2,
            num_return_sequences=beam_size,
            **tokenized_inputs,
        )
        .cpu()
        .detach()
        .numpy()
    )

    errors = tokenizer.batch_decode(tokenized_labels, skip_special_tokens=True)
    errors = [errors[i : i + beam_size] for i in range(0, len(errors), beam_size)]
    return errors


def predict_token_class(tokenizer: RobertaTokenizerFast, model: RobertaForTokenClassification, error: List[str], source: List[str]) -> List[List[int]]:
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


def predict_source_code(tokenizer: RobertaTokenizerFast, model: T5ForConditionalGeneration, error: List[str], source: List[str], beam_size=BEAM_SIZE) -> List[str]:
    tokenized_inputs = tokenizer(
        text=error,
        text_pair=source,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    tokenized_labels = (
        model.generate(
            num_beams=beam_size,
            no_repeat_ngram_size=2,
            num_return_sequences=beam_size,
            max_length=512,
            **tokenized_inputs,
        )
        .cpu()
        .detach()
        .numpy()
    )

    sources = tokenizer.batch_decode(tokenized_labels, skip_special_tokens=True)
    sources = [sources[i : i + beam_size] for i in range(0, len(sources), beam_size)]
    return sources


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

    def run(self, source_code: Union[str, List[str]], beam_size=BEAM_SIZE):
        if isinstance(source_code, str):
            source_code = [source_code]

        print("Predicting error description...")
        error_descriptions = predict_error_description(
            self.tokenizer_ed, self.model_ed, source_code, beam_size
        )

        print("Predicting token class...")
        token_classes = [[] for _ in source_code]
        for error_description in zip(*error_descriptions):
            token_class = predict_token_class(
                self.tokenizer_tc, self.model_tc, error_description, source_code
            )
            for i, tc in enumerate(token_class):
                token_classes[i].append(tc)
    
        print("Predicting source code...")
        new_sources = [[] for _ in source_code]
        for error_description in zip(*error_descriptions):
            new_source = predict_source_code(
                self.tokenizer_cg, self.model_cg, error_description, source_code, beam_size
            )
            for i, ns in enumerate(new_source):
                new_sources[i].append(ns)

        return error_descriptions, token_classes, new_sources


if __name__ == "__main__":
    session = Session()
    print(session.run("A = map(input().split())\nprint(A)"))
