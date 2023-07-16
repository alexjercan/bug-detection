# Hint

The hint generation experiments

Supported datasets

- alexjercan/AoC
- alexjercan/bugnet

Supported models:

- gpt-3.5-turbo

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

This is still WIP.
