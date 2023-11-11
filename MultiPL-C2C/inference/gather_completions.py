"""
This script produces completions from Codex, or any model that fudges the
completions API.

To run this script, create a file called model_keys.csv with the columns:

```
Label,Model,Key
My OpenAI Key,,sk-YOUR-OPENAI-KEY
A100,codegen,http://localhost:8080/v1/completions
A40,incoder,http://localhost:8080/v1/completions
```
   
OpenAI API keys will only work if you are in the Codex beta.

Then run:

python3 completions_codex.py --dir <TARGET_DIR> --max-samples=32

The max_samples argument should be a reasonable number. Too small, and
you won't be using your GPU very efficently. Too high, and you'll crash
a naive implementation.
"""
import csv
import pdb
from typing import Dict
import os

import openai_multimodel_multikey
import asyncio
import tqdm.asyncio
import argparse
import json
import sys
from pathlib import Path
import logging

from dataset_builder.utils import TOFILL_TOKEN, MULTI_INTERMEDIATE_GENERATION_FLAG

MAX_TO_GENERATE = 750 # 512

# problem is a dict. Relevant keys are "name" and "prompt".
async def process_problem_json(completion, problem, args, max_to_generate):
    target_dir_path = Path(args.target_dir)
    completions_path = target_dir_path / (problem["name"] + ".json")
    os.makedirs(target_dir_path, exist_ok=True)

    if completions_path.exists():
        with completions_path.open() as f:
            try:
                completion_results = json.load(f)
            except:
                print(f"json load error {f}")
    else:
        # Copy problem to completion_results
        completion_results = problem.copy()
        completion_results["completions"] = []

    # completion_results has the same keys as problem, and one extra key "completions".
    num_completions_required = args.limit_completions - len(completion_results["completions"])

    if num_completions_required < 1:
        return

    # nicer to have print message updating status of 20 beam w/ intermediate generation
    verbose = False
    if args.max_samples == 1 and isinstance(problem["translation_prompt"], list):
        verbose = True

    while num_completions_required > 0:
        num_samples = min(num_completions_required, args.max_samples)
        prompt, prompt_idx, num_incomplete_prompts = "", 0, 0
        if "translation_prompt" not in problem:
            prompt = problem["prompt"]
        elif isinstance(problem["translation_prompt"], str):
            prompt = problem["translation_prompt"] if TOFILL_TOKEN in completion_results["translation_prompt"] else completion_results["translation_prompt"]
        elif isinstance(problem["translation_prompt"], list):
            # if there are incomplete prompts, do 1 time intermedidate generation with multiple generations to
            # generate all different intermediate steps at once, then generate the rest 1 by 1
            num_incomplete_prompts = len([p for p in completion_results["translation_prompt"] if TOFILL_TOKEN in p])

            # once all intermediate steps are generated, generate the rest 1 by 1
            prompt_idx = len(completion_results["completions"]) % len(problem["translation_prompt"])
            prompt = completion_results["translation_prompt"][prompt_idx]
            if num_incomplete_prompts > 0:
                prompt = MULTI_INTERMEDIATE_GENERATION_FLAG+prompt
                num_samples = num_incomplete_prompts
            # num_samples will be number of intermediate steps needs to generate, but entire generation number will be 1
            # assert num_samples == 1, "with multi-intermediate step, can only generate 1 sample at a time"
        else:
            raise NotImplementedError("Do not recognize translation prompt data type")
        updated_prompt, completions = await completion(
            model=args.model,
            prompt=prompt,
            max_tokens=max_to_generate,
            temperature=args.temperature,
            n=num_samples,
            top_p=0.95,
            stop=[s for s in problem["stop_tokens"]] if problem["stop_tokens"] else None,
        )
        completion_results["completions"].extend(completions)
        if "translation_prompt" in problem and updated_prompt:
            if isinstance(problem["translation_prompt"], str) and TOFILL_TOKEN in completion_results["translation_prompt"]:
                # explicitely updating the translation prompt, otherwise it is only sometimes updated, weird
                completion_results["translation_prompt"] = updated_prompt
            elif isinstance(problem["translation_prompt"], list) and num_incomplete_prompts > 0:
                # if we have list of intermediate steps, only when it's not stored, we update the corresponding prompt
                completion_results["translation_prompt"][-num_incomplete_prompts:] = updated_prompt
                if verbose:
                    print(f"Finished intermediate for {problem['name']}: num_samples_required={num_completions_required}")
                num_samples = 1
        with completions_path.open("w") as f:
            f.write(json.dumps(completion_results, indent=2))

        num_completions_required -= num_samples
        if verbose:
            print(f"completed 1 for {problem['name']}: num_samples_required={num_completions_required}")


def configure_logging(args):
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if args.log_file:
        logging.basicConfig(
            filename=args.log_file, level=args.log_level, format=format_str
        )
    else:
        logging.basicConfig(level=args.log_level, format=format_str)


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--prompts-file", type=str, required=True, help="File of prompts")
    args.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="Directory to write completions to",
    )
    args.add_argument("--temperature", type=float, required=True)
    args.add_argument("--max-samples", type=int, required=True)
    args.add_argument("--model", type=str, required=True)
    args.add_argument("--limit-completions", type=int, default=200)
    args.add_argument("--log-file", type=str, default=None)
    args.add_argument("--log-level", type=str, default="INFO")
    args.add_argument(
        "--local-model",
        action="store_true",
        help="If set, --model is the name of a model file to load",
    )
    args = args.parse_args()
    args.process_function = process_problem_json
    return args


async def main(args):
    prompts_file = Path(args.prompts_file)
    if not prompts_file.exists():
        print("File does not exist: {}".format(prompts_file))
        sys.exit(1)

    if args.model == "davinci":
        args.model = "code-davinci-002"

    configure_logging(args)

    with prompts_file.open() as f:
        problems = json.load(f)

    if args.local_model:
        completions = __import__(args.model).completion
        for problem in problems:
            await args.process_function(
                completions, problem, args, max_to_generate=MAX_TO_GENERATE
            )
    else:
        # Load the model keys from the CSV file.
        with open("model_keys_azure.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        model_keys = [
            row["Key"] for row in rows if not row["Key"].startswith("http://")
        ]
        other_models = [
            (row["Model"], row["Key"])
            for row in rows
            if row["Key"].startswith("http://")
        ]
        async with openai_multimodel_multikey.MultiModelMultiKeyCompletion(
            model_keys, other_models
        ) as completions:
            problem_completions = [
                args.process_function(
                    completions.completion,
                    problem,
                    args,
                    max_to_generate=MAX_TO_GENERATE,
                )
                for problem in problems
            ]
            # await asyncio.gather(*problem_completions)
            pbar = tqdm.tqdm(total=len(problems))
            for f in asyncio.as_completed(problem_completions):
                value = await f
                pbar.set_description(value)
                pbar.update()

if __name__ == "__main__":
    asyncio.run(main(get_args()))
