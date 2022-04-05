import numpy as np

from transformers import (
    RobertaTokenizerFast,
    T5ForConditionalGeneration,
    RobertaForTokenClassification,
)


LIGHT_THEME = {"err_color": "red", "norm_color": "black", "ws_color": "lightgrey"}
DARK_THEME = {"err_color": "red", "norm_color": "white", "ws_color": "grey"}


def predict_error_description(tokenizer, model, source):
    tokenized_inputs = tokenizer(
        source, padding=True, truncation=True, return_tensors="pt"
    ).to(model.device)
    tokenized_labels = model.generate(**tokenized_inputs).cpu().detach().numpy()

    return tokenizer.batch_decode(tokenized_labels, skip_special_tokens=True)


def predict_token_class(tokenizer, model, error, source):
    if not isinstance(source, list):
        source = [source]
        error = [error]

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


def predict(tokenizer_ed, model_ed, tokenizer_tc, model_tc, source):
    error = predict_error_description(tokenizer_ed, model_ed, source)
    labels = predict_token_class(tokenizer_tc, model_tc, error, source)

    return error, labels


def color_source(
    source_code, mask, err_color="red", norm_color="black", ws_color="lightgrey"
):
    text = ""
    for i, char in enumerate(source_code):
        color = norm_color
        if char == " ":
            char = "•"
            color = ws_color
        if char == "\n":
            char = "↵\n"
            color = ws_color
        text += (
            f'<span style="color:{err_color if mask[i] == 1 else color};">{char}</span>'
        )
    return "<pre>" + text + "</pre>"


def run(source_code, tokenizer_ed, model_ed, tokenizer_tc, model_tc, theme=LIGHT_THEME):
    if isinstance(source_code, str):
        source_code = [source_code]

    error, labels = predict(tokenizer_ed, model_ed, tokenizer_tc, model_tc, source_code)
    source_code_html = [
        color_source(src, labels[i], **theme) for i, src in enumerate(source_code)
    ]
    error_html = [f"<pre>{err}</pre>" for err in error]

    result = []
    for src, err in zip(source_code_html, error_html):
        result.append(f"<h1>Source code</h1>{src}<h1>Error description</h1>{err}")

    return result


if __name__ == "__main__":
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

    source_code = """a = 1
A = A + 5 * a
print(A)
"""

    result = run(source_code, tokenizer_ed, model_ed, tokenizer_tc, model_tc)

    print("\n".join(result))
