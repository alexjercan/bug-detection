# CodeGen

The results on the `CodeGen` language model for the `BugNet` dataset. This also
includes `CodeT5`. You can switch between the two models using the `--model`
flag, the available models are `codegen` and `codet5`. You can also change the
prompting method from `simple` to `multishot` using the `--prompt` flag. The
`simple` method will create the prompt by appending "propose a fix for the bug"
to the source code. The `multishot` prompt will concatenate N (hardcoded to 3)
pairs of buggy/accepted source code files and then finally append the buggy
source code that the model has to predict.

## Quickstart

1. Make sure you have the BugNet dataset generated into `../input/bugnet`

2. Run the script and generate the model results

```console
python main.py --model codegen
```

The results will be generated in a csv file that is prefixed with the name of
the model (for example "codegen_results.csv").

3. Run Jupyter to see notebook examples

```console
jupyter notebook
```

## About CodeGen, CodeT5 and Results

This is still WIP.

