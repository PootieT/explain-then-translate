#!/usr/bin/env bash

ROOT=PATH/TO/REPO
export PYTHONPATH="${PYTHONPATH}:${ROOT}/MultiPL-C2C:${ROOT}/CodeGenMirror:${ROOT}/CodeGenMirror/codegen_sources/test_generation"
LANGS=(js)  # js cpp java ts php rb cs go pl r rs scala swift sh lua rkt jl d
SRC=py
for lang in "${LANGS[@]}"
do
#PAIRS=("js java" "cpp py" "java cpp" "js rkt" "cpp lua" "java jl") # "rkt java" "jl js" "d cpp" "lua rkt" "rkt jl" "d lua" "jl d"
#for i in "${PAIRS[@]}"
#do
#  set -- $i # convert the "tuple" into the param args $1 $2...
#  lang=$2
#  SRC=$1
  echo "translating from ${SRC} to ${lang}..."
  EXP_NAMES=(\
  "${SRC}-${lang}/humaneval-${SRC}-${lang}-PTremove-completion" \
  "${SRC}-${lang}/humaneval-${SRC}-${lang}-PTremove-MTexplain-completion" \
  "${SRC}-${lang}/humaneval-${SRC}-${lang}-PTremove-MTexplain-lbl-completion" \
  "${SRC}-${lang}/humaneval-${SRC}-${lang}-PTremove-MTexplain-lbl-simp-completion" \
  "${SRC}-${lang}/humaneval-${SRC}-${lang}-PTremove-4shot-completion" \
  "${SRC}-${lang}/humaneval-${SRC}-${lang}-PTremove-MTexplain-4shot-completion" \
  "${SRC}-${lang}/humaneval-${SRC}-${lang}-PTremove-MTexplain(0shot)-4shot-completion" \
  "${SRC}-${lang}/humaneval-${SRC}-${lang}-PTremove-MTexplain-lbl-4shot-completion" \
  )
  for EXP_NAME in "${EXP_NAMES[@]}"
  do
    OUTPUT_DIR="dump/${EXP_NAME}"
    NUM_GENERATIONS=20

    echo "======== experiment ${EXP_NAME}, num-generation:${NUM_GENERATIONS} ========="
    python inference/gather_completions.py \
      --prompts-file translation_prompts/$EXP_NAME.json \
      --target-dir $OUTPUT_DIR \
      --temperature 0.2 \
      --max-samples $NUM_GENERATIONS \
      --model gpt-3.5-turbo \
      --limit-completions $NUM_GENERATIONS

    ## local eval
    cd evaluation/src
    python main.py --dir "../../${OUTPUT_DIR}" --output-dir "../../${OUTPUT_DIR}" --recursive
    cd ../..

    # podman eval
    #podman run --rm --network none -v "./${OUTPUT_DIR}:/${OUTPUT_DIR}:rw" multipl-e-eval --dir "/${OUTPUT_DIR}" --output-dir "/${OUTPUT_DIR}" --recursive

    python3 src/single_experiment_pass_k.py --temperature 0.2 $OUTPUT_DIR
    python3 src/single_experiment_error_types.py --temperature 0.2 $OUTPUT_DIR
  done
done