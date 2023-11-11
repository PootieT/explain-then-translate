import pdb

import datasets
import argparse
import gzip
import json
import importlib
from pathlib import Path
from tqdm import tqdm
import sys
import pandas as pd
import torch

from inference.chatgpt_utils import cleanup_completion_simple
from inference.openai_multimodel_multikey import prompt_incomplete, get_incomplete_translation_prompt, \
    get_intermediate_stops, fill_translation_prompt

DATASET_REVISION = "bf4f3c31a1e0a164b7886c9eb04f82534edf4ce9"


def from_remote_dataset(args):
    problems = datasets.load_dataset(
        "nuprl/MultiPL-E", f"{args.root_dataset}-{args.lang}",
        revision=DATASET_REVISION
    )
    problems = problems["test"]
    start_index = args.input_start_index if args.input_start_index is not None else 0
    stop_index = min(
        len(problems),
        start_index + args.input_limit
        if args.input_limit is not None
        else len(problems),
    )
    problems = problems.select(range(start_index, stop_index))
    return problems


def from_local_dataset(args):
    with open(args.dataset, "r") as f:
        problems_list = json.load(f)
        start_index = args.input_start_index if args.input_start_index is not None else 0
        stop_index = min(
            len(problems_list),
            start_index + args.input_limit
            if args.input_limit is not None
            else len(problems_list),
        )

        #problems = datasets.Dataset.from_list(problems_list[start_index:stop_index])
        problems = datasets.Dataset.from_pandas(pd.DataFrame(problems_list[start_index:stop_index]))
    return problems


def main():
    args = argparse.ArgumentParser()

    args.add_argument(
        "--output-dir",
        type=str,
        help="Directory in which to place JSON files with completions. The default is root_dataset-lang-model_name-temperature-reworded",
    )

    args.add_argument(
        "--output-dir-prefix",
        type=str,
        help="Prefix for the output directory"
    )

    args.add_argument('--use-local', action="store_true", help="Use this flag when running from local prompts.")

    # Reuired when use local is passed
    args.add_argument(
        "--dataset", type=str, required="--use-local" in sys.argv,
        help="The local dataset in JSON format to get from this computer."
    )
    # Only required when use local is not passed
    args.add_argument(
        "--lang", type=str, required="--use-local" not in sys.argv, help="Target language for completions"
    )
    args.add_argument(
        "--root-dataset", type=str, required="--use-local" not in sys.argv, help="either mbpp or humaneval"
    )
    args.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="either incoder or codegen. To add a new model, copy and modify codegen.py",
    )
    args.add_argument("--temperature", type=float, required=True)
    args.add_argument(
        "--input-start-index",
        type=int,
        help="Index into the dataset. If omitted, starts from the beginning",
    )
    args.add_argument(
        "--input-limit", type=int, help="Number of items to process from the dataset"
    )
    args.add_argument("--completion-limit", type=int, default=200)
    args.add_argument(
        "--batch-size", type=int, default=16, help="Number of completions to batch"
    )
    args = args.parse_args()

    model = importlib.import_module(args.model_name)

    if args.output_dir is None:
        args.output_dir = (
            f"{args.root_dataset}-{args.lang}-{model.name}-{args.temperature}-reworded"
        ) if not args.use_local else (
            f"{args.dataset.split('/')[-1].split('.')[0]}-{model.name}-{args.temperature}-reworded"
        )

    if args.output_dir_prefix is not None:
        args.output_dir = f"{args.output_dir_prefix}/{args.output_dir}"

    exp_dir = Path(args.output_dir)
    if not exp_dir.exists():
        exp_dir.mkdir()

    if args.use_local:
        problems = from_local_dataset(args)
    else:
        problems = from_remote_dataset(args)

    for problem in tqdm(problems, unit="problems"):
        # NOTE(arjun): This is a litte hack to delay loading the model, so that we fail faster.
        # problem_filename = exp_dir / f"{problem['name']}.json.gz"
        problem_filename = exp_dir / f"{problem['name']}.json"
        if problem_filename.exists():
            # with gzip.open(problem_filename, "rt") as f:
            with open(problem_filename, "r") as f:
                existing = json.loads(f.read())
            completions = existing["completions"]
            problem["translation_prompt"] = existing["translation_prompt"]
        else:
            completions = []

        if len(completions) > args.completion_limit:
            # Not strictly necessary, but avoid a pointless rewriting of the file with no changes.
            continue

        # if prompts are not complete (i.e. need to generate explanation before translation), do intermediate completion
        prompt = problem["translation_prompt"]
        while prompt_incomplete(prompt, problem):
            incomplete_prompt = get_incomplete_translation_prompt(prompt)
            intermediate_stops = get_intermediate_stops(incomplete_prompt, problem)
            intermediate_completion = model.completions(
                prompt=incomplete_prompt,
                max_tokens=750,  # 512 normally
                temperature=args.temperature,
                n=1,
                top_p=0.95,
                stop=intermediate_stops,
            )
            prompt = fill_translation_prompt(prompt, intermediate_completion[0])
        problem["translation_prompt"] = prompt

        for _ in tqdm(
                range(len(completions), args.completion_limit, args.batch_size),
                unit="completions",
        ):
            # new_completions = model.completions(
            #     prompt=problem["translation_prompt"],
            #     max_tokens=512,  # 512 normally
            #     temperature=args.temperature,
            #     n=args.batch_size,
            #     top_p=0.95,
            #     stop=problem["stop_tokens"],
            # )
            new_completions, bs = [], args.batch_size
            while len(new_completions) < args.batch_size:
                try:
                    raw_completions = model.completions(
                        prompt=problem["translation_prompt"],
                        max_tokens=512,  # 512 normally
                        temperature=args.temperature,
                        n=bs,
                        top_p=0.95,
                        stop=problem["stop_tokens"],
                    )
                    new_completions.extend([cleanup_completion_simple(c, problem["translation_prompt"])
                                            for c in raw_completions])
                except RuntimeError as e:
                    print(f"GPU OOM error {e}: \nhalving batch size from {bs} to {bs//2} and trying again..")
                    bs = bs // 2
                    torch.cuda.empty_cache()
                    if bs == 0:
                        raise Exception("Batch size cannot be zero. Sequence too long, consider smaller model, larger "
                                        "GPU, or loading model on multiple GPUs")

            completions.extend(new_completions)

        result_json = problem.copy()
        result_json["completions"] = completions
        # with gzip.open(problem_filename, "wt") as f:
        with open(problem_filename, "w") as f:
            json.dump(result_json, f)


if __name__ == "__main__":
    main()
