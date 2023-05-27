# Bug Detection and Repair

Bug Detection and Repair Dataset with Exploratory Data Analysis examples and
demo applications for Bug Detection and Repair algorithm.

## Content

Source code related tasks for machine learning have become important with the
large need of software production. In this project our main goal is to create a
dataset for bug detection and repair and also provide examples of code repair
algorithms.

This repository contains the scripts used for dataset generation. We have also
added some exploratory data analysis notebooks for the generated datasets. The
`codenetpy` folder contains the script used to generate the `CodeNetPy`
dataset, the `bugnet` folder contains the script used to generate the `BugNet`
dataset, and the `repair-pipeline` folder contains the demo applications for
the models trained only on `CodeNetPy`.

To install the dependencies for development create a venv:

```console
python -m venv .venv
source .venv/bin/activate
make install
```

1. To generate the `CodeNetPy` dataset see [codenetpy](./codenetpy/)

2. To run the repair pipeline see [repair-pipeline](./repair-pipeline/)

3. To generate the `BugNet` dataset see [bugnet](./bugnet/)

4. To visualize the results of `Codex` on the `BugNet` dataset see [codex](./codex/)

5. To visualize the results of the `CodeGen` model on the `BugNet` dataset see [codegen](./codegen/)

6. To visualize the results of the description generation on the `BugNet` dataset see [description](./description//)

7. To visualize the `AoC` dataaset see [aoc-dataset](./aoc-dataset/)
