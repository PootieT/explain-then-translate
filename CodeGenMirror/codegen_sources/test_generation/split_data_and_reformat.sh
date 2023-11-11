#!/bin/bash

langs="python"
obfuscate="False"
add_spans="False"
project_root="/home/shypula/CodeGenMirror"
evaluation_set_directory="/nobackup/users/shypula/github_data_compressed/github_public_licensed_data/ST_output_all_aggregate/online_ST_files/test_dataset/" # path to the directory containing the original files with valid / test data

overwrite="True"
testcase_prop="0.05"
process_evaluation_set=True
make_offline_training_dataset=True
make_online_training_dataset=True

bpe_codes_path="$project_root/data/bpe/cpp-java-python/codes"
bpe_vocab_path="$project_root/data/bpe/cpp-java-python/vocab"


original_data_root="/nobackup/users/shypula/github_data_compressed/github_public_licensed_data/"
new_data_root="/nobackup/users/shypula/model_data"

root_to_plain_data="ST_output_all_aggregate"
root_to_obfuscated_data="ST_output_aggregate_all_obfuscated"
root_to_span_data="ST_output_aggregate_all_spans"

for root_to_data in $root_to_plain_data  $root_to_obfuscated_data  $root_to_span_data; do
  path_to_data_in="$original_data_root/$root_to_data"
  path_to_data_out="$new_data_root/$root_to_data"

  echo "now copying the directory $path_to_data_in to $path_to_data_out"

  echo '''python3 $project_root/codegen_sources/test_generation/subset_testcases_and_df.py \
          --root_dir_in $path_to_data_in \
          --root_dir_out $path_to_data_out \
          --overwrite $overwrite \
          --testcase_prop $testcase_prop'''

  python3 $project_root/codegen_sources/test_generation/subset_testcases_and_df.py \
          --root_dir_in $path_to_data_in \
          --root_dir_out $path_to_data_out \
          --overwrite $overwrite \
          --testcase_prop $testcase_prop


  echo "copied the directory $path_to_data_in to $path_to_data_out"

  input_df_path="${path_to_data_out}/translations_results/test_results/test_results_python_df.csv" # path to the .csv with the python translations and test results
  echo "reformat data at $path_to_data_out and df at $input_df_path"
  echo "obfuscate is $obfuscate and spans are set to $add_spans"

  echo '''python3 $project_root/codegen_sources/test_generation/reformat_data.py \
          --langs $langs \
          --obfuscate $obfuscate \
          --add_spans $add_spans \
          --input_df_path $input_df_path \
          --project_root $project_root \
          --output_directory $path_to_data_out \
          --process_evaluation_set $process_evaluation_set \
          --evaluation_set_directory $evaluation_set_directory \
          --make_offline_training_dataset $make_offline_training_dataset \
          --make_online_training_dataset $make_online_training_dataset \
          --bpe_codes_path $bpe_codes_path \
          --bpe_vocab_path $bpe_vocab_path'''

  python3 $project_root/codegen_sources/test_generation/reformat_data.py \
          --langs $langs \
          --obfuscate $obfuscate \
          --add_spans $add_spans \
          --input_df_path $input_df_path \
          --project_root $project_root \
          --output_directory $path_to_data_out \
          --process_evaluation_set $process_evaluation_set \
          --evaluation_set_directory $evaluation_set_directory \
          --make_offline_training_dataset $make_offline_training_dataset \
          --make_online_training_dataset $make_online_training_dataset \
          --bpe_codes_path $bpe_codes_path \
          --bpe_vocab_path $bpe_vocab_path

    ## the first time in the loop we
    if [ $obfuscate == "False" ]; then
      obfuscate="True"
    else
      obfuscate="False"
      add_spans="True"
    fi

done
