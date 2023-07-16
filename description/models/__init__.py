from args import MODEL_CHATGPT

from .chatgpt import ChatGPTPipeline
from .model import Pipeline


def make_pipeline(checkpoint: str) -> Pipeline:
    if checkpoint == MODEL_CHATGPT:
        return ChatGPTPipeline()

    raise ValueError(f"Unknown checkpoint: {checkpoint}")


__all__ = [
    "make_pipeline",
    "Pipeline",
]
