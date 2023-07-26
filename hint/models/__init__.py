from args import MODEL_CHATGPT, MODEL_LLAMA2_HF

from .chatgpt import ChatGPTPipeline
from .llama2_hf import Llama2HfPipeline
from .model import Pipeline


def make_pipeline(checkpoint: str) -> Pipeline:
    if checkpoint == MODEL_CHATGPT:
        return ChatGPTPipeline()

    if checkpoint == MODEL_LLAMA2_HF:
        return Llama2HfPipeline()

    raise ValueError(f"Unknown checkpoint: {checkpoint}")


__all__ = [
    "make_pipeline",
    "Pipeline",
]
