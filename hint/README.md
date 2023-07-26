# Hint

The hint generation experiments

Supported datasets

- alexjercan/AoC
- alexjercan/bugnet

Supported models:

- gpt-3.5-turbo (ChatGPT)
- llama2-hf (llama2-70B-chat)

## Quickstart

1. Run the script and generate the model results

```console
python main.py --dataset alexjercan/bugnet --model gpt-3.5-turbo
```

The results will be generated into a huggingface dataset in a file named
"./input/hint/evaluation_data_{data_name}_{model_name}.data" (for example
./input/hint/evaluation_data_bugnet_gpt-3.5-turbo.data).

2. Run Jupyter to see notebook examples

```console
jupyter notebook
```

## About the results

The metrics that I have used in this experiment are exact match and bleu score.

First I have generated the bug description by using the given model. Then I
have computed the bug type. Basically we can look at the generated text, and if
it is talking about *input* function like `input` or `read`, etc. we can assume
that it is an input related bug, similarly for `output`. Lastly I have
considered that all other bugs are related to the general `algorithm`. Then, we
can do the same but on the diff between the passing and failing source code.
This will give us what part of the file was changed: `input`, `output` or
`algorithm`. Finally we can compute exact match to check if the model at least
had some intuition about the type of bug.

For the other metric I have also used the generated bug description as
prediction. To compute the reference I have used again the diff between the
failing submission and the passing submission, and also the error description
(this is only available for bugnet though). I have concatenated the diff and
the error description and thus built the reference text. Finally I computed the
bleu score between the predictions and references. My idea is that a good
prediction would contain some words related to the diff or the error
description, which gives us a higher bleu score.

For the AoC dataset using 2 predictions per problem with ChatGPT I have got a
bleu score of 0.0116 +- 0.0062 on average. The exact match of the bug type was
0.6777 +- 0.3154 on average. I have also observed a high correlation between
the two metrics. When the bleu score was really low, the exact match was also
low (for example bleu 0.005 and match 0.2333), and when the bleu score was
larger, the exact match was also larger (for example bleu 0.023 and match
0.9333).

I have also repeated the experiment for the bugnet dataset and have used 2
predictions with ChatGPT. The average bleu score was 0.0429 +- 0.0028 and the
average exact match on the bug type was 0.58 +- 0.0282. In this experiment I
couldn't notice the same correlation between the two metrics as I did with the
AoC dataset. One possible theory is that AoC contains only `algorithm` bugs,
where the bugnet dataset contains all 3 types. With `input` bugs being the most
common.

| Model            | Dataset | exact match | bleu     |
|------------------|---------|-------------|----------|
| gpt-3.5-turbo*   | AoC     | 0.6777      | 0.0116   |
| gpt-3.5-turbo*   | bugnet  | 0.58        | 0.0429   |
| llama2-70B-chat  | AoC     | 1.0         | 0.0      |
| llama2-70B-chat  | bugnet  | 0.26        | 0.0      |

- \* I have used the chatgpt version [ChatGPT July 20
  Version](https://help.openai.com/en/articles/6825453-chatgpt-release-notes)
