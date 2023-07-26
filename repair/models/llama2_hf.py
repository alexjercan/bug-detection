import logging
import json
import os

import time
from gradio_client import Client

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


def make_prompt_simple(source: str) -> str:
    return f"<s>[INST] <<SYS>>\n{SYSTEM}\n<</SYS>>\n\n{source}[/INST]"


class Llama2HfPipeline(Pipeline):
    def __init__(self, num_sequences=2):
        self.num_sequences = num_sequences

    def __call__(self, example, **kwargs):
        fail = example["fail"]
        prompt = make_prompt_simple(fail)

        client = Client("https://ysharma-explore-llamav2-with-tgi.hf.space/")

        try:
            result = client.predict(prompt, api_name="/chat")
        except Exception as e:
            logging.error(e)
            result = ""

        llama2_results = [try_parse_content(result) for _ in range(self.num_sequences)]

        time.sleep(1)

        return llama2_results
