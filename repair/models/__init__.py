from args import MODEL_CHATGPT, MODEL_CODEGEN2, MODEL_LLAMA2_HF

from .chatgpt import ChatGPTPipeline
from .codegen2 import CodeGen2Pipeline
from .llama2_hf import Llama2HfPipeline
from .model import Pipeline


def make_pipeline(checkpoint: str) -> Pipeline:
    if checkpoint == MODEL_CODEGEN2:
        return CodeGen2Pipeline()

    if checkpoint == MODEL_CHATGPT:
        return ChatGPTPipeline()

    if checkpoint == MODEL_LLAMA2_HF:
        return Llama2HfPipeline()

    raise ValueError(f"Unknown checkpoint: {checkpoint}")


__all__ = [
    "make_pipeline",
    "Pipeline",
]
