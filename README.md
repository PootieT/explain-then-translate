# explain-then-translate
Official repo for EMNLP 2023 paper "Explain-then-Translate: An Analysis on Improving Program Translation with Self-generated Explanations"
Our code translation dataset `MultiPL-C2C` is also released here for benefiting future research in this direction.

- Open Review [link](https://openreview.net/forum?id=uyUO80sbm0)

## What is the take-away?
We propose 2-stage Chain-of-Thought (CoT) like prompting technique for program translation: we ask models to explain the source programs first before translating.

<img src="https://github.com/PootieT/explain-then-translate/blob/main/docs/prompt_main.png" width="350">

We tested across 3 type of explanations of different level of abstraction, across 18 Python-to-X directions, and 18 X-to-X directions. 
This simple technique improves translation performance across 4 models of different sizes we tested (GPT-3.5, Llama2CodeInstruct-34B, CodeGen2-16B, and CodeGen2-1B).
We found better explanations results in better translation(i.e CodeGen2-1B translates better with GPT-3.5 explanations than self-generated explanations)
We additionally propose simple heuristics to improve translation by sampling multiple self-explanations and selecting the one with highest heuristic score. Our heuristics, however, still leaves huge gap (60% relatively) to oracle trial that suggests valuable future direction.
We also found that explanations improve difficult-to-translate examples, suggesting that a pre-selection difficulty predictor can be employed to select programs to explain, improving accuracy in addition to efficiency.
Lastly, we release MultiPL-C2C, a code-translation version of MultiPL-E, for future code translation research.

### Repo lineage

We heavily modified our dataset from [MultiPL-E](https://github.com/nuprl/MultiPL-E). However since the divergence of our evaluation system many scripts are not compatible anymore. If you are interested in updating this repo to make it compatible with theirs, we welcome PRs! That being said, MultiPL-E is indoubitably a better engineered repository, friendly to evaluating on cluster, with docker. For inspirations on how to evaluate our dataset with larger scale, please study their repo and [tutorial](https://nuprl.github.io/MultiPL-E/).

## Environment setup:

Make a new python3.10 environment
```shell
conda create python=3.10 --name code310
```
Install the following dependencies
```shell
pip3 install aiohttp numpy tqdm pytest datasets torch transformers
./CodeGenMirror/install_env.sh
```
The second line installs dependencies from CodGen repo, which are used for post-processing
Python, Java, and C++ completions. If translation is not generating in those directions, 
the second line of installation is not needed.

## How to generate prompts:

All translation prompts should be available in `translation_prompts` directory. To generate these prompt files, we ran:

```shell
cd dataset_builder
python3 all_prepare_translation_prompts.py --trail py_x
python3 all_prepare_translation_prompts.py --trail x_x
cd ..
```
This script generates translation prompts for all language directions, few-shot cases, explanation types by invoking

```shell
python3 prepare_prompts_json.py --lang humaneval_to_<TGT_LANG>.py \
        --prompt-terminology remove \
        --originals ./datasets/originals \
        --source-program <SRC_LANG> \
        --shot 0 \
        --multiturn_prompt explain \
        --output ./translation_prompts
```
For some files, it would require you to sample some generation first. For example, all `Python-to-X` directions uses 
cached explanation, so generate one direction, sample an explanation from each program, then you can generate the rest 
of `Python-to-X` directions.

If you want to customize your prompts, we recommend digging into `python3 prepare_prompts_json.py`.

As an example, if you want to use a different intermediate prompt (e.g. `chain of thought` instead of `explain`), 
simply run:
```shell
python3 prepare_prompts_json.py --lang humaneval_to_<TGT_LANG>.py \
        --prompt-terminology remove \
        --originals ./datasets/originals \
        --source-program <SRC_LANG> \
        --shot 0 \
        --multiturn_prompt CoT \
        --output ./translation_prompts
```

To generate all 19*19 translation direction files, you can run
```shell
cd dataset_builder
python3 all_prepare_translation_prompts.py --trail all
cd ..
```

## How to sample programs:

We used Azure completion endpoint to query GPT-3.5. To follow our setup, you can run something like this:
```shell
python inference/gather_completions.py \
  --prompts-file translation_prompts/<EXP_NAME>.json \
  --target-dir <OUTPUT_DIR> \
  --temperature 0.2 \
  --max-samples 20 \
  --model gpt-3.5-turbo \
  --limit-completions 20
```

If you want to use a huggingface model locally, you can do something like this
```shell
python inference/__main__.py \
      --model-name codegen21b \
      --output-dir <OUTPUT_DIR> \
      --output-dir-prefix local_codegen21b \
      --use-local \
      --dataset translation_prompts/$EXP_NAME.json \
      --temperature 0.2 \
      --completion-limit 20 \
      --batch-size 1
```
checkout `inference/codegen21b.py` file to understand how to extend to new models.

We prioritize translation precision, so we use a low temperature to optimize **pass@1** (n=20). 
For **pass@10** or **pass@100** you need much larger n. See bottom section of MultiPL-E 
[tutorial](https://nuprl.github.io/MultiPL-E/tutorial.html)

## How to evaluate:

Assuming you have all 19 languages install locally, you can run:

```shell
cd evaluation/src
python main.py --dir "../../<OUTPUT_DIR>" --output-dir "../../<OUTPUT_DIR>" --recursive
cd ../..
python analysis/collect_completion_results.py --dir <OUTPUT_DIR>
```

However, installing all 19 languages can be a little annoying, so we are working on building a container for 
execution just the same way as MultiPL-E [tutorial]. This is coming soon.

## MultiPL-C2C

The gold programs from our dataset in languages other than Python are sampled from GPT-3.5. Because it is sampled, not every problem has a gold solution in each language. Therefore, if during the your sampling you are able to obtain more completion results, feel free to send PR requests and we will update our `datasets/humaneval_multi` directory.

To facilitate aggregating gold solutions from completion folders, you can run
```shell

```
It collects the shortest passing completion for each problem for each target langauge.

## To cite our work:

```
@inproceedings{tang2023explain,
  title={Explain-then-Translate: An Analysis on Improving Program Translation with Self-generated Explanations},
  author={Tang, Zilu and Agarwal, Mayank and Shypula, Alex and Wang, Bailin and Wijaya, Derry and Chen, Jie and Kim, Yoon},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  year={2023},
  url={https://aclanthology.org/2023.findings-emnlp.196/}
}
```
