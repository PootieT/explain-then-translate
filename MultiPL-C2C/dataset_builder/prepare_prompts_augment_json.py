"""
This script prepares all prompts for a particular language as YAML files (one
per benchmark). The scripts completions_*.py will then update each file with
completions from an LLM.

To run this script:

1. mkdir ../datasets/LANGUAGE-keep-MODEL

  where MODEL is either davinci or incoder.

2. python3 prepare_prompts_yaml.py --lang LANGUAGE --target-dir ../datasets/LANGUAGE-keep-MODEL --doctests keep

  This will create lots of YAML files in TARGET-DIR. You should commit these files to the repository.

3. Now run either completions_codex.py or completions_incoder.py.


Estimate of how big each YAML file gets:

- length of prompt + completion = 2048 tokens
- Each token is ~4 characters
- 200 samples per prompt
- (2048 * 4 * 200) / 1024 / 1024 = 1.5 MB of data.

This ignores the tests cases, but it should be compact enough.

"""

import os
import argparse
import csv
import sys
from typing import Union, Dict

import regex as re

import asyncio

import tqdm

from dataset_builder.utils import bool_flag
from generic_translator import list_originals, translate_prompt_and_tests, get_stop_from_translator
from pathlib import Path
import json

from inference import openai_multimodel_multikey
from inference.gather_completions import process_problem_json

MAX_TO_GENERATE = 1024  # max for program only is 512, but with comments, program can get long


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--output", type=str, required=False, default="../datasets/humaneval-comments-inline", help="Target JSON file"
    )

    args.add_argument(
        "--comment",
        type=str,
        default="inline",
        help="What kind of comments to add. One of [inline, docstring, docstring_w_test, all]",
    )

    args.add_argument(
        "--few_shot_file",
        type=str,
        default=None,
        help="Path to few shot file (.json) containing the style of comments to be added",
    )

    args.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Number of few-shot examples to provide",
    )
    args.add_argument("--originals", type=str, default="../datasets/originals-with-cleaned-doctests")

    # model completion args
    args.add_argument("--temperature", type=float, default=0.2, help="generation temperature")
    args.add_argument("--max-samples", type=int, default=1)
    args.add_argument("--model", type=str, default="gpt-3.5-turbo")
    args.add_argument("--limit-completions", type=int, default=1)
    args.add_argument(
        "--local-model",
        action="store_true",
        help="If set, --model is the name of a model file to load",
    )

    args = args.parse_args()
    return args


def post_process_python_generation(output_dir, original_dir):
    for f in os.listdir(output_dir):
        if f.endswith(".json") and not f.endswith(".results.json"):
            res = json.load(open(f"{output_dir}/{f}"))
            scaffold = open(f"{original_dir}/{res['name']}.py").read()
            start = scaffold.find("### Canonical solution below ###") + len("### Canonical solution below ###") + 1
            end = scaffold.find("### Unit tests below ###") - 1
            new_code = scaffold[:start] + res["completions"][0] + "\n\n" + scaffold[end:]
            open(f"{output_dir}/{res['name']}.py", "w").write(new_code)


def extract_source_program(file_path):
    entry_point = re.search("([^0-9]+_\d+)_(.+).py", file_path.name).group(2)
    reading_prompt = True
    reading_tests = False
    reading_source_program = False
    prompt_buffer = []
    tests_buffer = []
    source_program_buffer = []
    with open(file_path) as f:
        for line in f:
            if "### Canonical solution below ###" in line:
                reading_prompt = False
                reading_source_program = True
            if "### Unit tests below ###" in line:
                reading_tests = True
                reading_source_program = False
                continue
            if "def test_check():" in line:
                break

            if reading_prompt:
                prompt_buffer.append(line)
            if reading_tests:
                tests_buffer.append(line)
            if reading_source_program:
                source_program_buffer.append(line)
    # prompt = "".join(prompt_buffer)
    tests = "".join(tests_buffer)
    stop = 0
    while not prompt_buffer[stop].startswith("def"):
        stop += 1
    source_lines = prompt_buffer[:stop + 1] + source_program_buffer[1:]
    source_code = "".join(source_lines).strip()
    prompt_str = f"### original python code\n\n{source_code}\n\n" \
                 f"### commented python code\n\n{prompt_buffer[stop]}"
    return prompt_str, tests


def add_augmentation_prompt(prompt: str, comments, few_shot_files, shots) -> Union[str, Dict]:
    match comments:
        case "inline":
            instruction = "Can you annotate the following python program with inline comments? The goal is to make this piece of code easy to understand by other code readers."
        case "docstring":
            instruction = "Can you annotate the following python program with docstring summarizing the purpose of the program? The goal is to make this piece of code easy to understand by other code readers."
        case "docstring_w_test":
            instruction = "Can you annotate the following python program with docstring summarizing the purpose of the program? Make sure to include example unit tests in the docstring as well of what an example input and output would look like. The goal is to make this piece of code easy to understand by other code readers."
        case "all":
            instruction = "Can you annotate the following python program with inline comments and docstring summarizing the purpose of the program? Make sure to include example unit tests in the docstring as well of what an example input and output would look like. The goal is to make this piece of code easy to understand by other code readers."
        case other:
            raise NotImplementedError

    prompt = instruction + "\n\n" + prompt

    if few_shot_files is not None:
        few_shot_data = [json.loads(l) for l in open(few_shot_files, "r").readlines()]
        assert all([d["content"].startswith(instruction) for d in few_shot_data if d["role"]=="user"])
        few_shot_data = few_shot_data[:shots]
        few_shot_data.append({"role": "user", "content":prompt})
        prompt = few_shot_data

    return prompt


async def main(args):
    args.target_dir = args.output
    if args.comments not in ["inline", "docstring", "docstring_w_test", "all"]:
        print(f"Invalid comments option: {args.comments}")
        sys.exit(1)
    if args.few_shot_file is not None or args.shots > 0:
        assert os.path.isfile(args.few_shot_file)

    os.makedirs(args.output, exist_ok=True)

    problems = []
    for original in list_originals(args.originals).values():
        original_name = original.name.split(".")[0]
        print(f"Processing {original_name}...")

        result = extract_source_program(original)
        if result is None:
            print(f"Skipping {original_name}")
            continue

        (prompt, tests) = result
        augment_prompt = add_augmentation_prompt(prompt, args.comments, args.few_shot_file, args.shots)
        problem = {
            "name": original_name,
            "language": "py",
            "prompt": prompt,
            "original": str(original.absolute()),
            "tests": tests,
            "stop_tokens": ["\n###", "\nclass"],
            "translation_prompt": augment_prompt
        }
        problems.append(problem)

    problems = [p for p in problems if not os.path.exists(f"{args.output}/{p['name']}.json")]

    if args.local_model:
        completions = __import__(args.model).completion
        for problem in problems:
            await process_problem_json(
                completions, problem, args, max_to_generate=MAX_TO_GENERATE
            )
    else:
        # Load the model keys from the CSV file.
        with open("../model_keys_azure.csv") as f:
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
                process_problem_json(
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

    post_process_python_generation(args.output, args.originals)

if __name__ == "__main__":
    args = get_args()
    args.comments = "docstring"
    args.temperature = 0.2
    args.shots = 0
    args.output = f"../datasets/originals-comments-{args.comments}-s{args.shots}"
    # args.few_shot_file = "../few_shot_prompts/inline_comments_few_shot_prompts.jsonl"
    asyncio.run(main(args))
