# BugNet

The BugNet Dataset with Exploratory Data Analysis examples.

## Quickstart

1. Download the CodeNet dataset and generate BugNet into `../input/bugnet`

```console
python main.py
```

2. Run Jupyter to see notebook examples

```console
jupyter notebook
```

## About the Dataset

The generated dataset can be found on Kaggle. Dataset can be found
[here](https://www.kaggle.com/datasets/alexjercan/bugnet).

The dataset is based on the CodeNet project and contains python and C++ code
submissions for online coding competitions. The data is obtained by selecting
consecutive attempts of a single user that resulted in fixing a buggy
submission. Thus the data is represented by code pairs and annotated by the
diff and error of each changed instruction. We have already tokenized all the
source code files and kept the same format as in the original dataset.

The upgrade made compared to `CodeNetPy` is that we only keep one line errors.
This means that the task of bug detection and repair will be easier to manage.
We also removed all the files that fail on linters, so that we are focusing
only on bugs that cannot be identified easily.

This is still WIP.

## Acknowledgements

CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding
Tasks

## Limitations

- The problem p02974 is causing the script to crash on my pc because of OOM
