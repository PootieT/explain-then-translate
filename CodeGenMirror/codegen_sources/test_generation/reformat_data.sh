#!/bin/bash

langs="python"
obfuscate="True"
add_spans="False"
input_df_path="" # path to the .csv with the python translations and test results
project_root="~/CodeGenMirror"
output_directory="" # path to the directory where everything will be put into
process_evaluation_set=True
evaluation_set_directory="" # path to the directory containing the original files with valid / test data
make_offline_training_dataset=True
make_online_training_dataset=True
bpe_codes_path="$project_root/data/bpe/cpp-java-python/codes"
bpe_vocab_path="$project_root/data/bpe/cpp-java-python/vocab"

python3 $project_root/codegen_sources/test_generation/reformat_data.py \
        --langs $langs \
        --obfuscate $obfuscate \
        --add_spans $add_spans \
        --input_df_path $input_df_path \
        --project_root $project_root \
        --output_directory $output_directory \
        --process_evaluation_set $process_evaluation_set \
        --evaluation_set_directory $evaluation_set_directory \
        --make_offline_training_dataset $make_offline_training_dataset \
        --make_online_training_dataset $make_online_training_dataset \
        --bpe_codes_path $bpe_codes_path \
        --bpe_vocab_path $bpe_vocab_path