import openai
import time
from typing import Any, Dict, List

from .model import Pipeline


def make_chatgpt_prompt_simple(source: str) -> str:
    return f"{source}"


def chatgpt_inference(example: Dict[str, Any], num_sequences: int) -> List[str]:
    fail = example["fail"]
    prompt = make_chatgpt_prompt_simple(fail)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        n=num_sequences,
        messages=[
            {
                "role": "system",
                "content": "Respond with code only. "
                "The response will be used in a compiler, "
                "so do NOT include any type of comments. "
                "Propose code to fix the bug.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    assert isinstance(response, dict), "Failed to get a response from OpenAI."

    chatgpt_results = [r["message"]["content"] for r in response["choices"]]

    time.sleep(1)

    return chatgpt_results


class ChatGPTPipeline(Pipeline):
    def __init__(self, num_sequences=2):
        self.num_sequences = num_sequences

    def __call__(self, example, **kwargs):
        return chatgpt_inference(example, self.num_sequences)
