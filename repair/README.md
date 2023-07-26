# Repair

The repair experiments.

Supported datasets

- alexjercan/AoC
- alexjercan/bugnet

Supported models:

- gpt-3.5-turbo (ChatGPT)
- Salesforce/codegen2-1B
- llama2-hf (llama2-70B-chat)

## Quickstart

1. Run the script and generate the model results

```console
python main.py --dataset alexjercan/bugnet --model gpt-3.5-turbo
```

The results will be generated into a huggingface dataset in a file named
"evaluation_data_{data_name}_{model_name}.data" (for example
evaluation_data_bugnet_gpt-3.5-turbo.data).


2. Run Jupyter to see notebook examples

```console
jupyter notebook
```

## About the results

The metrics that I have used for this experiments are `pass@k`, `exact match
accuracy` and `bug type match`.

For the `pass@k` score I have used k=1,2. So I have generated 2 predictions per
source code file and then tested if the first one runs (the one with the
highest probability), and then if any of the 2 pass the tests. The `exact
match` is self explanatory, I just tested if the prediction is exactly the same
as the source code from the ground truth. This is not the best metric since we
can fix code in many different ways, but it is interesting to see if any of the
models will find a fix that is in the dataset. A high `exact match` score might
also be an indicator that the model was pretrained on the tested dataset. For
the `bug type` score, I have tested to see if the prediction will attempt to
change the same type of instruction as the ground truth fix. This is not an
indicator of correctness, but again, it is interesting to see if the model
attempted to fix the instruction that is actually buggy, and did not
halucinate. Basically input, output or general algorithm bugs. See
[hint](/hint) for more on bug type score.

For ChatGPT, on the AoC dataset I have gotten a `pass@1` of 0.0333, a `pass@2`
of 0.0666. The `exact match` was 0, so I guess that the AoC data was not used
in training ChatGPT (which is pretty obvious since I have created AoC in May
2023). For the `bug type` I got a 0.6 score, which means that 60% of the times,
ChatGPT attempted to change something relevant to the bug.

For ChatGPT, on the bugnet dataset I

For Llama2, on the AoC dataset I have used the
[space](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI)
provided as an example. It is the 70B parameter model finetuned for chat. I got
scores of 0.0, except for bug type where I got 1.0. This is not that great,
maybe it can be improve somehow.

| Model            | Dataset | pass@1 | pass@2 | exact match | bug type |
|------------------|---------|--------|--------|-------------|----------|
| gpt-3.5-turbo*   | AoC     | 0.3333 | 0.6666 | 0.0         | 0.6      |
| gpt-3.5-turbo*   | bugnet  | 0.795  | 0.86   | 0.0         | 0.75     |
| llama2-70B-chat  | AoC     | 0.0    | 0.0    | 0.0         | 1.0      |
| llama2-70B-chat  | bugnet  | 0.0    | 0.0    | 0.0         | 0.26     |

- \* I have used the chatgpt version [ChatGPT July 20
  Version](https://help.openai.com/en/articles/6825453-chatgpt-release-notes)
