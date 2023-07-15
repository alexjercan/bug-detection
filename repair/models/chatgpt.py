import time
import openai
from typing import Dict, List
from .model import Pipeline


def make_chatgpt_prompt_simple(source: str) -> str:
    return f"{source}\nPropose code to fix the bug\n"


def chatgpt_inference(example: Dict[str, List], num_sequences: int) -> List[List[str]]:
    # TODO: use num_sequences

    predictions = []
    for fail in example["fail"]:
        prompt = make_chatgpt_prompt_simple(fail)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a senior developer."},
                {"role": "user", "content": prompt},
            ]
        )
        chatgpt_result = response["choices"][0]["message"]["content"]

        predictions.append([chatgpt_result])

        time.sleep(1)

    return predictions


class ChatGPTPipeline(Pipeline):
    def __init__(self, num_sequences=2):
        self.num_sequences = num_sequences

    def __call__(self, examples, **kwargs):
        return chatgpt_inference(examples, self.num_sequences)
