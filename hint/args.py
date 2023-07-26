import logging

import argparse
from dataclasses import dataclass

DATASET_BUGNET = "alexjercan/bugnet"
DATASET_AOC = "alexjercan/AoC"

MODEL_CHATGPT = "gpt-3.5-turbo"
MODEL_LLAMA2_HF = "llama2-hf"


@dataclass
class Options:
    """Options for the script."""

    debug: bool = False
    dataset: str = DATASET_BUGNET
    subset: str = "Python"
    checkpoint: str = MODEL_CHATGPT
    num_sequences: int = 2
    split: str = "test"
    loglevel: int = logging.INFO


def parse_args() -> Options:
    levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }

    parser = argparse.ArgumentParser("Evaluate a hint generation model")

    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        dest="loglevel",
        choices=list(levels.keys()),
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_BUGNET,
        choices=[DATASET_BUGNET, DATASET_AOC],
        help="Dataset to use",
    )
    parser.add_argument("--subset", type=str, default="Python", help="Subset to use")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=MODEL_CHATGPT,
        choices=[MODEL_CHATGPT, MODEL_LLAMA2_HF],
        help="Checkpoint to use",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=2,
        help="Number of sequences to generate",
    )
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate on")

    args = parser.parse_args()

    return Options(
        debug=args.debug,
        dataset=args.dataset,
        subset=args.subset,
        checkpoint=args.checkpoint,
        num_sequences=args.num_sequences,
        split=args.split,
        loglevel=levels[args.loglevel],
    )
