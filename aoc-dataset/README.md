# AoC Dataset

![checker](https://github.com/alexjercan/bug-detection/actions/workflows/aoc-dataset.yml/badge.svg)

A collection of submissions for the Advent of Code challenge. This repository
contains both passing and failing submissions.

You can find the dataset contents on [huggingface](https://huggingface.co/datasets/alexjercan/AoC).

## About the Dataset

This dataset is inspired by [HumanEval](https://github.com/openai/human-eval)

The source code used to generate the dataset can be found on [GitHub](https://github.com/alexjercan/bug-detection/tree/master/aoc-dataset)

A collection of submissions for the Advent of Code challenge.
This repository contains both passing and failing submissions.

This dataset is similar to [BugNet](https://huggingface.co/datasets/alexjercan/bugnet),
however it is meant to be used as an evaluation dataset.

The resulting dataset file will be a csv with the following columns:
- `year`: Used to identify the submission
- `day`: Used to identify the submission
- `part`: Used to identify the submission
- `fail`: The initial (buggy) source code formatted (`black`)
- `pass`: The modified (accepted) source code formatted (`black`)
- `change`: The change that was made (`replace`, `insert`, `delete`)
- `i1`: Start of the change in the buggy source (the line; starting with 1)
- `i2`: End of the change in the buggy source (not inclusive; for insert we have i1 == i2)
- `j1`: Start of the change in the accepted source (the line; starting with 1)
- `j2`: End of the change in the accepted source (not inclusive; for delete we have j1 == j2)
- `test`: The test case that can be used to evaluate the submission.

