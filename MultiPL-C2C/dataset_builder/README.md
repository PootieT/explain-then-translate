# MultiPL-C2C CDataset Builder

## Introduction

This directory contains the code to build the MultiPL-C2C dataset from source
Python programs. You only need to work with this code if you're trying to
add a new dataset or programming language to MultiPL-C2C.

If your goal is to evaluate a model with MultiPL-E-C2C, you can ignore this
code and use the pre-built dataset on the HuggingFace hub:

https://huggingface.co/datasets/zilu-peter-tang/MultiPL-C2C

## Requirements

Python 3.10+

## Usage

To builds everything (in our experiments):

```shell
python3 all_prepare_translation_propmts.py --trial x_x
python3 all_prepare_translation_propmts.py --trial py_x
```
See `all_prepare_translation_propmts.py` for details on caching explanations 
and re-using across Python-X directions (for comparison purpose)

If you just want to generate all pairs of translation directions without cache:
```shell
python3 all_prepare_translation_propmts.py --trial all
```

