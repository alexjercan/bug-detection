import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List

from .model import Pipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_example_batched(example: Dict[str, List]) -> Dict[str, List]:
    original_srcs = [src.splitlines(keepends=False) for src in example["fail"]]
    changed_srcs = [src.splitlines(keepends=False) for src in example["pass"]]
    i1s = example["i1"]
    i2s = example["i2"]
    j1s = example["j1"]
    j2s = example["j2"]

    prefixes = [
        "\n".join(original_src[:i1]) for original_src, i1 in zip(original_srcs, i1s)
    ]
    suffixes = [
        "\n".join(original_src[i2:]) for original_src, i2 in zip(original_srcs, i2s)
    ]
    changes = [
        "\n".join(changed_src[j1:j2])
        for changed_src, j1, j2 in zip(changed_srcs, j1s, j2s)
    ]

    return {
        "text": [
            prefix + "\n<mask_1>\n" + suffix + "<|endoftext|>" + "<sep>" + "<mask_1>"
            for prefix, suffix in zip(prefixes, suffixes)
        ],
        "label": changes,
    }


def codegen2_inference(
    example: Dict[str, Any],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num_sequences: int,
) -> List[str]:
    example = format_example_batched(example)

    text = str(example["text"])

    tokenized_input = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    tokenized_input = tokenized_input.to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **tokenized_input,
            max_new_tokens=512,
            num_beams=num_sequences,
            num_return_sequences=num_sequences,
            early_stopping=True,
        )

    # Predictions is a list of shape (batch_size, num_sequences)
    predictions = []
    for i in range(num_sequences):
        mask_prediction = tokenizer.decode(output_ids[i])[len(text) :].split("<eom>")[0]
        predictions.append(
            text.replace("<mask_1>", mask_prediction).split("<|endoftext|>", maxsplit=1)[
                0
            ]
        )

    return predictions


class CodeGen2Pipeline(Pipeline):
    def __init__(self, num_sequences=2):
        checkpoint = "Salesforce/codegen2-1B"

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, trust_remote_code=True, revision="main"
        )
        model = model.to(DEVICE)

        self.tokenizer = tokenizer
        self.model = model
        self.num_sequences = num_sequences

    def __call__(self, example: Dict[str, Any], **kwargs) -> List[str]:
        return codegen2_inference(
            example, self.model, self.tokenizer, self.num_sequences
        )
