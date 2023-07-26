import os
import json

import openai
import time

from .model import Pipeline

path = os.path.dirname(os.path.abspath(__file__))
prompt_path = os.path.join(path, "chatgpt.prompt.txt")

with open(prompt_path, "r", encoding="utf-8") as f:
    SYSTEM = f.read()


def try_parse_content(content: str) -> str:
    try:
        repair = json.loads(content)["source"]
        return repair
    except json.decoder.JSONDecodeError:
        return content


def make_chatgpt_prompt_simple(source: str) -> str:
    return f"{source}"


class ChatGPTPipeline(Pipeline):
    def __init__(self, num_sequences=2):
        self.num_sequences = num_sequences

    def __call__(self, example, **kwargs):
        fail = example["fail"]
        prompt = make_chatgpt_prompt_simple(fail)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            n=self.num_sequences,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM,
                },
                {"role": "user", "content": prompt},
            ],
        )
        assert isinstance(response, dict), "Failed to get a response from OpenAI."

        chatgpt_results = [
            try_parse_content(r["message"]["content"]) for r in response["choices"]
        ]

        time.sleep(1)

        return chatgpt_results
