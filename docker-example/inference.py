import numpy as np

from difflib import SequenceMatcher
from transformers import (
    RobertaTokenizerFast,
    T5ForConditionalGeneration,
    RobertaForTokenClassification,
)


LIGHT_THEME = {"norm_color": "black", "ws_color": "lightgrey"}
DARK_THEME = {"norm_color": "white", "ws_color": "grey"}

BEAM_SIZE = 5


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
        "alexjercan/codet5-base-buggy-code-repair"
    )
    model_cg = T5ForConditionalGeneration.from_pretrained(
        "alexjercan/codet5-base-buggy-code-repair"
    )

    return tokenizer_ed, model_ed, tokenizer_tc, model_tc, tokenizer_cg, model_cg


def predict_error_description(
    tokenizer, model, source: list, beam_size=BEAM_SIZE
) -> list:
    tokenized_inputs = tokenizer(
        source, padding=True, truncation=True, return_tensors="pt"
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

    return tokenizer.batch_decode(tokenized_labels, skip_special_tokens=True)


def predict_token_class(tokenizer, model, error: list, source: list) -> list:
    tokenized_inputs = tokenizer(
        text=error, text_pair=source, padding=True, truncation=True, return_tensors="pt"
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

        all_labels.append(labels)

    return all_labels


def predict_source_code(
    tokenizer, model, error: list, source: list, beam_size=BEAM_SIZE
) -> list:
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

    return tokenizer.batch_decode(tokenized_labels, skip_special_tokens=True)


def predict(
    tokenizer_ed,
    model_ed,
    tokenizer_tc,
    model_tc,
    tokenizer_cg,
    model_cg,
    source: list[str],
    beam_size=BEAM_SIZE,
) -> (list, list, list):
    error = predict_error_description(tokenizer_ed, model_ed, source, beam_size)
    source = [src for src in source for _ in range(beam_size)]

    labels = predict_token_class(tokenizer_tc, model_tc, error, source)
    source_code = predict_source_code(tokenizer_cg, model_cg, error, source, beam_size)

    return error, labels, source_code


def generate_char_mask(original_src: str, changed_src: str) -> list:
    s = SequenceMatcher(None, original_src, changed_src)
    opcodes = [x for x in s.get_opcodes() if x[0] != "equal"]

    original_labels = np.zeros_like(list(original_src), dtype=np.int32)
    for _, i1, i2, _, _ in opcodes:
        original_labels[i1 : max(i1 + 1, i2)] = 1

    return original_labels.tolist()


def color_source(
    source_code: str,
    mask: list,
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


def run(
    source_code,
    tokenizer_ed,
    model_ed,
    tokenizer_tc,
    model_tc,
    tokenizer_cg,
    model_cg,
    theme=LIGHT_THEME,
    beam_size=BEAM_SIZE,
):
    if isinstance(source_code, str):
        source_code = [source_code]

    error, labels, new_source_code = predict(
        tokenizer_ed,
        model_ed,
        tokenizer_tc,
        model_tc,
        tokenizer_cg,
        model_cg,
        source_code,
        beam_size=beam_size,
    )
    source_code = [src for src in source_code for _ in range(beam_size)]
    source_code_html = [
        color_source(src, labels[i], **theme)
        for i, src in enumerate(source_code)
        for _ in range(beam_size)
    ]
    error_html = [f"<pre>{err}</pre>" for err in error for _ in range(beam_size)]

    source_code = [src for src in source_code for _ in range(beam_size)]
    new_source_code_html = [
        color_source(
            new_src, generate_char_mask(new_src, src), accent_color="green", **theme
        )
        for new_src, src in zip(new_source_code, source_code)
    ]

    result = []
    for src, err, new_src in zip(source_code_html, error_html, new_source_code_html):
        result.append(
            f"<h1>Source code</h1>{src}<h1>Error description</h1>{err}<h1>Repaired code</h1>{new_src}"
        )

    return result
