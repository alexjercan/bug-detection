# Bug Detection and Repair

![yeet](https://github.com/alexjercan/bug-detection/actions/workflows/checks.yml/badge.svg)

Bug Detection and Repair Dataset with Exploratory Data Analysis examples and
demo applications for Bug Detection and Repair algorithm.

## Content

Source code related tasks for machine learning have become important with the
large need of software production. In this project our main goal is to create a
dataset for bug detection and repair and also provide examples of code repair
algorithms.

This repository contains the scripts used for dataset generation. We have also
added some exploratory data analysis notebooks for the generated datasets. The
`bugnet` folder contains the script used to generate the *BugNet* dataset. The
`repair-pipeline` folder contains the demo applications for the models trained
only on the Python code from *BugNet*. The `aoc-dataset` folder contains the
source code used to generate the *AoC* dataset. The `description` folder
contains the source code used to generate the description in natural language
of the bugs. The `repair` folder contains the source code used to evaluate
different models on the data that we collected.

To install the dependencies for development create a venv:

```console
python -m venv .venv
source .venv/bin/activate
make install
```

1. To run the repair pipeline see [repair-pipeline](./repair-pipeline/)

2. To generate the `BugNet` dataset see [bugnet](./bugnet/)

3. To visualize the `AoC` dataaset see [aoc-dataset](./aoc-dataset/)

4. To visualize the results of the description generation on the `BugNet` (of `AoC`) dataset see [description](./description//) -- upgrade

5. To visualize the result of the repair generation on the `BugNet` (or `AoC`) dataset see [repair](./repair/)
