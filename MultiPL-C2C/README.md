# Explain-then-Translate: an analysis on improving program translation with self-generated context

## Disclaimer

The repo is largely adapted from [MultiPl-E](https://github.com/nuprl/MultiPL-E). Any comments/names mentioned in code 
are kept as is from the original repo. 

This repo will be cleaned further before open-sourcing for the final version

## Prompt File Generation

To generate prompt files for experiments Python-to-X and X-to-X, simply run 
```bash
python all_prepare_translation_prompts.py
```

For reproducibility, we also include a here a [link]() to the generated prompts we used for our experiments

To customize your own prompt generation, see `dataset_builder/prepare_prompts_json.py`

## Completion Generation and Evaluation

For detailed information on completion generation and evaluation, please refer to original MultiPL-E's 
[readme](docs/multiple_original_readme.md) 

We mostly use `experiments/run_translate_codex.sh` and `run_eval.sh`

## Explanation Selection

See `inference/rerank_multi_prompts_json.py`

## Ablations

See `dataset_builder/prepare_alternative_prompts.py`


