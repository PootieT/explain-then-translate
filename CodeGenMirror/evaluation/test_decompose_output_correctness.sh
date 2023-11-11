### evaluate generate_candidate function output correctness, result should preserve
# computational accuracy (100%)
project_dir="/home/tangpihai/Project"
extraction_method="no_extract"

results_file="${project_dir}/decompose-and-translate/data/transcoder_evaluation_gfg/${extraction_method}/python.json"
outfolder="${project_dir}/decompose-and-translate/dump/${extraction_method}/python/decomposition_validation"
script_folder="${project_dir}/CodeGenMirror/data/transcoder_evaluation_gfg"

mkdir -p $outfolder

echo "python3 ${project_dir}/decompose-and-translate/evaluation/test_outputs_correctness.py \
  --results_file $results_file \
  --outfolder $outfolder \
  --script_folder $script_folder\
  --roberta_mode true"

python3 ${project_dir}/generate_candidate-and-translate/evaluation/test_outputs_correctness.py \
  --results_file $results_file \
  --outfolder $outfolder \
  --script_folder $script_folder \
  --roberta_mode true

