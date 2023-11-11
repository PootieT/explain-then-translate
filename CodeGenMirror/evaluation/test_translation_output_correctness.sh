### evaluate translated function output correctness
project_dir="/root/to/project"
extraction_method="no_extract"
src_lang="java"
tgt_lang="python"
#model="transcoder"
model="codex"
dataset="transcoder_all_train_dev" # transcoder_evaluation_gfg, synthetic_blocks, transcoder_failed_train, transcoder_all_train
retry_mismatching_types=true
back_translate=false

#if $back_translate
#then
#  results_file="${project_dir}/decompose-and-translate/dump/${dataset}/${extraction_method}/${src_lang}/${src_lang}-${tgt_lang}-${src_lang}/translation_output_${model}.json"
#  outfolder="${project_dir}/decompose-and-translate/dump/${dataset}/${extraction_method}/${src_lang}/${src_lang}-${tgt_lang}-${src_lang}/translation_validation_${model}"
#else
#  results_file="${project_dir}/decompose-and-translate/dump/${dataset}/${extraction_method}/${tgt_lang}/${src_lang}-${tgt_lang}/translation_output_${model}.json"
#  outfolder="${project_dir}/decompose-and-translate/dump/${dataset}/${extraction_method}/${tgt_lang}/${src_lang}-${tgt_lang}/translation_validation_${model}"
#fi
#script_folder="${project_dir}/CodeGenMirror/data/${dataset}"

results_file="${project_dir}/decompose-and-translate/dump/${dataset}/${tgt_lang}/${src_lang}-${tgt_lang}/translation_output_${model}.json"
outfolder="${project_dir}/decompose-and-translate/dump/${dataset}/${tgt_lang}/${src_lang}-${tgt_lang}/translation_validation_${model}"
script_folder="${project_dir}/CodeGenMirror/data/transcoder_all_train"

mkdir -p $outfolder

echo "python3 ${project_dir}/decompose-and-translate/evaluation/test_outputs_correctness.py \
  --results_file $results_file \
  --outfolder $outfolder \
  --script_folder $script_folder\
  --roberta_mode false\
  --retry_mismatching_types true"

python3 ${project_dir}/decompose-and-translate/evaluation/test_outputs_correctness.py \
  --results_file $results_file \
  --outfolder $outfolder \
  --script_folder $script_folder \
  --retry_mismatching_types $retry_mismatching_types

