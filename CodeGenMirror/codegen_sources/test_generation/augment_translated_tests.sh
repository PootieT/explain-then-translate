#!/bin/bash

PROJ_ROOT="/home/shypula/CodeGenMirror"
ROOTDIR=$1

path_to_translated_tests_json=$ROOTDIR/"online_ST_files/translated_tests.json"
path_to_translated_tests_df=$ROOTDIR/"test_results_python_df.csv"
output_path=$ROOTDIR/"online_ST_files/augmented_translated_tests.json"

python3 $PROJ_ROOT/codegen_sources/test_generation/augment_translated_tests.py \
        --path_to_translated_tests_json $path_to_translated_tests_json \
        --path_to_translated_tests_df $path_to_translated_tests_df \
        --output_path $output_path
