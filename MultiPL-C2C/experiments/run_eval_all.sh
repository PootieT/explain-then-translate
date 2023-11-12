#!/usr/bin/env bash

ROOT=PATH/TO/REPO
export PYTHONPATH="${PYTHONPATH}:${ROOT}/MultiPL-C2C:${ROOT}/CodeGenMirror:${ROOT}/CodeGenMirror/codegen_sources/test_generation"

MODEL=codegen216b
# Evaluate all programs first
cd evaluation/src
python main.py --dir "../../dump_${MODEL}" --output-dir "../../dump_${MODEL}" --recursive
cd ../..

# Collect them in csv file for analysis
python analysis/collect_completion_results.py --dir "dump_${MODEL}"