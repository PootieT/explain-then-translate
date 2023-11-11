# Explain-then-Translate: an analysis on improving program translation with self-generated context

## Disclaimer

The repo is largely adapted from [MultiPl-E](https://github.com/nuprl/MultiPL-E). Any comments/names mentioned in code 
are kept as is from the original repo. 

## Prompt File Generation

If you simply want to evaluate models on translation performance, you may
skip this step and use existing prompts in `translation_prompts`. If you want
to customize your prompts however, you may generate prompt files for 
experiments Python-to-X and X-to-X by running
```bash
python dataset_builder/all_prepare_translation_prompts.py
```

To customize your own prompt generation, see `dataset_builder/prepare_prompts_json.py`

See more details in [tutorial](docs/tutorial.md)

## Completion Generation and Evaluation

You can evaluate either locally (assuming 19 languages installed), or through a docker
container, which we provide. For detailed information on completion generation and 
evaluation, please refer to [tutorial](docs/tutorial.md)

## Explanation Selection

See `inference/rerank_multi_prompts_json.py`

## Ablations

See `dataset_builder/prepare_alternative_prompts.py`




