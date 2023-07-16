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
                "content": "Respond with a description in natural language. "
                "You have to explain the possible error that can arise to the user.",
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
