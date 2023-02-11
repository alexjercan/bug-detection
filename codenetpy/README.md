# CodeNetPy

The CodeNetPy Dataset with Exploratory Data Analysis examples.

## Quickstart

1. Download the CodeNet dataset and generate CodeNetPy into `../input/codenetpy`

```console
python main.py
```

2. Run Jupyter to see notebook examples

```console
jupyter notebook
```

## About the Dataset

The generated dataset can be found on Kaggle. Dataset can be found
[here](https://www.kaggle.com/datasets/alexjercan/codenetpy).

The dataset is based on the CodeNet project and contains python code
submissions for online coding competitions. The data is obtained by selecting
consecutive attempts of a single user that resulted in fixing a buggy
submission. Thus the data is represented by code pairs and annotated by the
diff and error of each changed instruction. We have already tokenized all the
source code files and kept the same format as in the original dataset.

The dataset file contains source code pairs in json format as a list of
objects. Each object contains the "original\_src": source code of the buggy
submission file, the "changed\_src": source code of the accepted submission,
"problem\_id" identifier for the problem (there are multiple submissions for the
same problem), "original\_id" and "changed\_id" identifiers for the submissions,
"language": programming language (Python), "filename\_ext": extension of the
source code file (.py), "original\_status": status of the original submission on
the online competition (mostly runtime errors), "returncode": the return code
on the example input and output for the original submission, "error\_class":
parsed error of the original submission on the example input,
"error\_class\_extra": same as error\_class but with extra information, "error":
the error string as it appears in the console, "output": the output of the run,
if there was one.

### Acknowledgements

CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding
Tasks
