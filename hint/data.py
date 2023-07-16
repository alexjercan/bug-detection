from args import DATASET_AOC, DATASET_BUGNET
from datasets import Dataset, DatasetDict, load_dataset


def make_dataset(dataset_path: str, subset: str, split: str) -> Dataset:
    assert dataset_path in [
        DATASET_AOC,
        DATASET_BUGNET,
    ], f"Unknown dataset {dataset_path}."
    assert (dataset_path == DATASET_AOC and split == "train") or (
        dataset_path == DATASET_BUGNET
    ), "Only the `train` split of AoC is supported."

    dataset = load_dataset(dataset_path, subset)
    assert isinstance(dataset, DatasetDict), "Expected a DatasetDict."
    evaluation_data = dataset[split]

    if dataset_path == DATASET_AOC:
        evaluation_data = evaluation_data.add_column(  # type: ignore
            "language", ["Python" for _ in evaluation_data["pass"]]
        )

    return evaluation_data
