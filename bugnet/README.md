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

The generated dataset can be found on Huggingface. Dataset can be found
[here](https://huggingface.co/datasets/alexjercan/bugnet).

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

The script will generate the dataset using the steps:

1. Clean the problems (remove the ones with malformed metadata)
2. Generate submission pairs

The resulting dataset file will be a csv with the following columns:
- `problem_id`: The id of the problem, matches with the id from Project_CodeNet
- `language`: The programming language of the submission (`Python` or `C++`)
- `original_status`: The status of the initial submission (`TLE`, `MLE`, anything that is not `Accepted`)
- `fail`: The initial (buggy) source code formatted (`black` or `clang-fromat`)
- `pass`: The modified (accepted) source code formatted(`black` or `clang-format`
- `change`: The change that was made (`replace`, `insert`, `delete`)
- `i1`: Start of the change in the buggy source
- `i2`: End of the change in the buggy source
- `j1`: Start of the change in the accepted source
- `j2`: End of the change in the accepted source
- `error`: The error that was obtained running the buggy source code on the input/output examples
- `stderr`: The full output of stderr of running the buggy source code on the input/output examples
- `stdout`: The full output of stdout of running the buggy source code on the input/output examples
- `description`: The problem statement in html format
- `input`: The input for the test case
- `output`: The output for the test case

## Dependencies

To be able to generate the dataset you will require C++ and Python tools.

- `g++`: g++ (Ubuntu 12.2.0-3ubuntu1) 12.2.0
- `clang-tidy`: Ubuntu LLVM version 15.0.6
- `clang-format`: Ubuntu clang-format version 15.0.6
- `Python`: Python 3.10.7
- `black`: black, 23.1.0 (compiled: yes)
- `flake8`: 6.0.0 (mccabe: 0.7.0, pycodestyle: 2.10.0, pyflakes: 3.0.1) CPython 3.10.7 on Linux

These are the tools that are required. The scipt should work with newer versions aswell.

## Acknowledgements

CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding
Tasks

## Limitations

- The problem p02974 is causing the script to crash on my pc because of OOM.
  This happens because `black` cannot handle the `s237078692` and `s717600459`
  submissions. The user that made the solution was very smart and created a
  400K file. These are the funny files
  `Project_CodeNet/data/p02974/Python/s717600459.py`
  `Project_CodeNet/data/p02974/Python/s237078692.py` for anyone that is
  curious.
