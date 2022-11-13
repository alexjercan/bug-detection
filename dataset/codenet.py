import io
import os
import random
import zipfile
import urllib.request

import numpy as np

from transformers import PreTrainedTokenizerBase
from datasets import load_dataset, DatasetDict
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Callable


CODENETPY_URL = (
    "https://github.com/alexjercan/bug-detection/releases/download/1.0/codenetpy.zip"
)


def _download(root: str):
    print(f"Downloading codenetpy from {CODENETPY_URL} to {root}")

    os.makedirs(root, exist_ok=True)

    with urllib.request.urlopen(CODENETPY_URL) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zip:
            zip.extractall(os.path.join(root, "codenetpy"))


def _try_load_dataset(root: str):
    try:
        dataset = load_dataset(
            "json",
            data_dir=os.path.join(root, "codenetpy"),
            data_files={
                "train": "codenetpy_train.json",
                "test": "codenetpy_test.json",
            },
            field="data",
        )
    except:
        print(f"Could not find dataset in specified location: {root}")
        _download(root)
        dataset = load_dataset(
            "json",
            data_dir=os.path.join(root, "codenetpy"),
            data_files={
                "train": "codenetpy_train.json",
                "test": "codenetpy_test.json",
            },
            field="data",
        )

    return dataset


def _tokenize_dataset(
    dataset: DatasetDict,
    tokenize_and_align_labels: Callable[[Dict], Dict],
    batch_size: int,
) -> DatasetDict:
    train_dataset = (
        dataset["train"]
        .filter(lambda example: example["returncode"] != 0)
        .train_test_split(test_size=0.1)
    )
    test_dataset = dataset["test"]

    train_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        batch_size=batch_size,
        remove_columns=train_dataset["train"].column_names,
    )
    test_dataset = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        batch_size=batch_size,
        remove_columns=test_dataset.column_names,
    )

    dataset["train"] = train_dataset
    dataset["test"] = test_dataset

    return dataset


def detection(
    root: str, tokenizer: PreTrainedTokenizerBase, batch_size: int = 4
) -> DatasetDict:
    def tokenize_and_align_labels(example: Dict) -> Dict:
        example = {
            "original_src": example["original_src"] + example["changed_src"],
            "error_class_extra": example["error_class_extra"]
            + ["Accepted" for _ in example["changed_src"]],
        }

        tokenized_inputs = tokenizer(
            example["original_src"], padding=True, truncation=True
        )
        tokenized_y = tokenizer(
            example["error_class_extra"], padding=True, truncation=True
        )

        tokenized_inputs["labels"] = tokenized_y.input_ids
        return tokenized_inputs

    dataset = _try_load_dataset(root)

    return _tokenize_dataset(dataset, tokenize_and_align_labels, batch_size=batch_size)


def localization(
    root: str, tokenizer: PreTrainedTokenizerBase, batch_size: int = 4
) -> DatasetDict:
    def generate_char_mask(original_src: str, changed_src: str) -> List[int]:
        s = SequenceMatcher(None, original_src, changed_src)
        opcodes = [x for x in s.get_opcodes() if x[0] != "equal"]

        original_labels = np.zeros_like(list(original_src), dtype=np.int32)
        for _, i1, i2, _, _ in opcodes:
            original_labels[i1 : max(i1 + 1, i2)] = 1

        return original_labels.tolist()

    def tokenize_and_align_labels(example: Dict) -> Dict:
        example = {
            "original_src": example["original_src"] + example["changed_src"],
            "changed_src": example["changed_src"] + example["changed_src"],
            "error_class_extra": example["error_class_extra"]
            + ["Accepted" for _ in example["changed_src"]],
        }

        y = [
            generate_char_mask(x_o, x_c)
            for (x_o, x_c) in zip(example["original_src"], example["changed_src"])
        ]
        tokenized_inputs = tokenizer(
            text=example["error_class_extra"],
            text_pair=example["original_src"],
            padding=True,
            truncation=True,
        )

        labels = np.zeros_like(tokenized_inputs.input_ids, dtype=np.int32) - 100
        for i, y_i in enumerate(y):
            for j, y_i_j in enumerate(y_i):
                idx = tokenized_inputs.char_to_token(i, j, sequence_index=1)
                if idx is None:
                    continue
                if labels[i, idx] == -100:
                    labels[i, idx] = y_i_j
                else:
                    labels[i, idx] |= y_i_j

        tokenized_inputs["labels"] = labels.tolist()
        return tokenized_inputs

    dataset = _try_load_dataset(root)

    return _tokenize_dataset(dataset, tokenize_and_align_labels, batch_size=batch_size)


def repair(
    root: str, tokenizer: PreTrainedTokenizerBase, batch_size: int = 4
) -> DatasetDict:
    max_source_length = 256
    max_target_length = 512

    def generate_char_mask(
        original_src: str, changed_src: str
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        s = SequenceMatcher(None, original_src, changed_src)
        opcodes = [x for x in s.get_opcodes() if x[0] != "equal"]

        original_labels = np.zeros_like(list(original_src), dtype=np.int32)
        intervals = []
        for _, i1, i2, _, _ in opcodes:
            original_labels[i1 : max(i1 + 1, i2)] = 1
            intervals.append((i1, max(i1 + 1, i2)))

        return original_labels.tolist(), intervals

    def tokenize_and_align_labels(example: Dict) -> Dict:
        example = {
            "original_src": example["original_src"],
            "changed_src": example["changed_src"],
            "error_class_extra": example["error_class_extra"],
            "masked_src": ["" for _ in example["original_src"]],
            "generated_src": ["" for _ in example["original_src"]],
        }

        y = [
            generate_char_mask(x_o, x_c)
            for (x_o, x_c) in zip(example["changed_src"], example["original_src"])
        ]
        tokenized_code = tokenizer(
            text=example["changed_src"],
            max_length=max_source_length,
            padding=True,
            truncation=True,
        )
        for i, (_, y_intervals) in enumerate(y):
            if not y_intervals:
                j1 = random.randint(0, len(example["changed_src"][i]) - 1)
                j2 = random.randint(j1, len(example["changed_src"][i]))
            else:
                j1, j2 = random.choice(y_intervals)

            idx1 = tokenized_code.char_to_token(i, j1)
            idx2 = tokenized_code.char_to_token(i, j2)
            j1 = tokenized_code[i].token_to_chars(idx1)[0] if idx1 is not None else j1
            j2 = tokenized_code[i].token_to_chars(idx2)[1] if idx2 is not None else j2

            example["masked_src"][i] = (
                example["changed_src"][i][:j1]
                + "<mask>"
                + example["changed_src"][i][j2:]
            )
            example["generated_src"][i] = example["changed_src"][i][j1:j2]

        tokenized_inputs = tokenizer(
            text=example["error_class_extra"],
            text_pair=example["masked_src"],
            max_length=max_source_length,
            padding=True,
            truncation=True,
        )
        tokenized_y = tokenizer(
            example["generated_src"],
            max_length=max_target_length,
            padding=True,
            truncation=True,
        )

        labels = np.array(tokenized_y.input_ids)
        labels[labels == tokenizer.pad_token_id] = -100

        tokenized_inputs["labels"] = labels.tolist()
        return tokenized_inputs

    dataset = _try_load_dataset(root)

    return _tokenize_dataset(dataset, tokenize_and_align_labels, batch_size=batch_size)


if __name__ == "__main__":
    from transformers import RobertaTokenizerFast

    tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-base")
    dataset = detection(root="./data", tokenizer=tokenizer)
    print(dataset)

    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
    dataset = localization(root="./data", tokenizer=tokenizer)
    print(dataset)

    tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-base")
    dataset = repair(root="./data", tokenizer=tokenizer)
    print(dataset)
