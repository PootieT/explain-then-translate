#!/usr/bin/env bash

ROOT=PATH/TO/REPO
export PYTHONPATH="${PYTHONPATH}:${ROOT}/MultiPL-C2C:${ROOT}/CodeGenMirror:${ROOT}/CodeGenMirror/codegen_sources/test_generation"
MODEL=codegen21b

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
    OUTPUT_DIR="${EXP_NAME}"
    NUM_GENERATIONS=20

    echo "======== experiment ${EXP_NAME}, num-generation:${NUM_GENERATIONS} ========="
    mkdir -p "dump_${MODEL}/${OUTPUT_DIR}"
    python inference/__main__.py \
    --model-name $MODEL \
    --output-dir $OUTPUT_DIR \
    --output-dir-prefix dump_$MODEL \
    --use-local \
    --dataset translation_prompts_$MODEL/$EXP_NAME.json \
    --temperature 0.2 \
    --completion-limit $NUM_GENERATIONS \
    --batch-size 20

    ## local eval
#    cd evaluation/src
#    python main.py --dir "../../${OUTPUT_DIR}" --output-dir "../../${OUTPUT_DIR}" --recursive
#    cd ../..
    #
    # podman eval (so far only C++ really needs it for 1 script 137_compare_one)
    #podman run --rm --network none -v "./${OUTPUT_DIR}:/${OUTPUT_DIR}:rw" multipl-e-eval --dir "/${OUTPUT_DIR}" --output-dir "/${OUTPUT_DIR}" --recursive

#    python3 src/single_experiment_pass_k.py --temperature 0.2 $OUTPUT_DIR
#    python3 src/single_experiment_error_types.py --temperature 0.2 $OUTPUT_DIR
  done
done