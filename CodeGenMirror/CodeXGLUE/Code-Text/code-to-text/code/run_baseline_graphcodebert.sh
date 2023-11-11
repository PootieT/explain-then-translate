lang=java #programming language
lr=$1 #was 5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=saved_models/baseline_graphcodebert_${lang}_lr${lr}
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
epochs=10
config_name=microsoft/graphcodebert-base \
model_name_or_path=microsoft/graphcodebert-base \
tokenizer_name=microsoft/graphcodebert-base \
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl

python run.py --do_train --do_eval --model_type roberta --config_name $config_name --model_name_or_path $model_name_or_path --tokenizer_name $tokenizer_name --train_filename $train_file --dev_filename $dev_file \
--output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size \
--eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs && \
echo "training over, running evaluation" && \
python run.py --do_test --model_type roberta  --config_name $config_name --model_name_or_path $model_name_or_path --tokenizer_name $tokenizer_name  --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file \
--output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size && \
echo "***** Eval results  *****"; python ../evaluator/evaluator.py $output_dir/dev.gold < $output_dir/dev.output ; \
echo "***** Test results  *****"; python ../evaluator/evaluator.py $output_dir/test_1.gold < $output_dir/test_1.output
