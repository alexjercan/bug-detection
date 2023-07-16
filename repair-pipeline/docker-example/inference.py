import logging

import numpy as np
from copy import copy
from transformers import (
    RobertaForTokenClassification,
    RobertaTokenizerFast,
    T5ForConditionalGeneration,
)
from typing import List, Tuple, Union

BEAM_SIZE = 5


def predict_error_description(
    tokenizer: RobertaTokenizerFast,
    model: T5ForConditionalGeneration,
    source: List[str],
    beam_size=BEAM_SIZE,
) -> List[List[str]]:
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


def predict_token_class(
    tokenizer: RobertaTokenizerFast,
    model: RobertaForTokenClassification,
    error: List[str],
    source: List[str],
) -> List[List[int]]:
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
        for j, _ in enumerate(tokenized_labels[i]):
            if tokenized_inputs.token_to_sequence(i, j) != 1:
                continue

            word_id = tokenized_inputs.token_to_word(i, j)
            cs = tokenized_inputs.word_to_chars(i, word_id, sequence_index=1)
            if cs.start == cs.end:
                continue
            labels[cs.start : cs.end] |= tokenized_labels[i, j]

        all_labels.append([int(la) for la in labels])

    return all_labels


def predict_masked_source_code_step(
    tokenizer: RobertaTokenizerFast,
    model: T5ForConditionalGeneration,
    error: str,
    tokens: List[int],
    source: str,
    beam_size=BEAM_SIZE,
) -> Tuple[int, int, List[str]]:
    ct, i1, i2 = 0, 0, 0
    for i, t in enumerate(tokens):
        if t == 1 and ct == 0:
            i1 = i
            ct = 1
        if t == 0 and ct == 1:
            i2 = i
            break

    tokenized_inputs = tokenizer(
        text=[error],
        text_pair=[source[:i1] + "<mask>" + source[i2:]],
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

    return (
        i1,
        i2,
        tokenizer.batch_decode(tokenized_labels, skip_special_tokens=True),
    )


def predict_masked_source_code(
    tokenizer: RobertaTokenizerFast,
    model: T5ForConditionalGeneration,
    error: str,
    tokens: List[int],
    source: str,
    beam_size=BEAM_SIZE,
) -> List[str]:
    building_tokens = [copy(tokens)]
    building_sources = [copy(source)]

    def should_break(building_tokens):
        for bt in building_tokens:
            if any(bt):
                return False

        return True

    while not should_break(building_tokens):
        new_building_sources = []
        new_building_tokens = []

        for bs, bt in zip(building_sources, building_tokens):
            i1, i2, options = predict_masked_source_code_step(
                tokenizer, model, error, bt, bs, beam_size
            )

            for option in options:
                new_building_sources.append(bs[:i1] + option + bs[i2:])
                new_building_tokens.append(bt[:i1] + [0 for _ in option] + bt[i2:])

        building_sources = new_building_sources
        building_tokens = new_building_tokens

    return building_sources


def predict_source_code(
    tokenizer: RobertaTokenizerFast,
    model: T5ForConditionalGeneration,
    errors: List[str],
    tokens: List[List[int]],
    sources: List[str],
    beam_size=BEAM_SIZE,
) -> List[List[str]]:
    new_sources = []
    for error, token, source in zip(errors, tokens, sources):
        new_sources.append(
            predict_masked_source_code(tokenizer, model, error, token, source, beam_size)
        )

    return new_sources


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
            "alexjercan/codet5-base-masked-buggy-code-repair"
        )
        self.model_cg = T5ForConditionalGeneration.from_pretrained(
            "alexjercan/codet5-base-masked-buggy-code-repair"
        )

    def run(
        self,
        source_code: Union[str, List[str]],
        beam_size_ed=BEAM_SIZE,
        beam_size_cg=BEAM_SIZE,
    ):
        if isinstance(source_code, str):
            source_code = [source_code]

        logging.info("Generating error description...")
        error_descriptions = predict_error_description(
            self.tokenizer_ed, self.model_ed, source_code, beam_size_ed
        )

        logging.info("Predicting token classes...")
        token_classes: List[List[List[int]]] = [[] for _ in source_code]
        for error_description1 in zip(*error_descriptions):
            token_class1 = predict_token_class(
                self.tokenizer_tc,
                self.model_tc,
                list(error_description1),
                source_code,
            )
            for i, tc in enumerate(token_class1):
                token_classes[i].append(tc)

        logging.info("Generating source code...")
        new_sources: List[List[List[str]]] = [[] for _ in source_code]
        for error_description2, token_class2 in zip(
            zip(*error_descriptions), zip(*token_classes)
        ):
            new_source = predict_source_code(
                self.tokenizer_cg,
                self.model_cg,
                list(error_description2),
                list(token_class2),
                source_code,
                beam_size_cg,
            )
            for i, ns in enumerate(new_source):
                new_sources[i].append(ns)

        logging.info("Done.")

        return error_descriptions, token_classes, new_sources
