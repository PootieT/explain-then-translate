import datasets
import argparse
import gzip
import json
import importlib
from pathlib import Path
from tqdm import tqdm
import sys

DATASET_REVISION = "bf4f3c31a1e0a164b7886c9eb04f82534edf4ce9"    

def from_remote_dataset(args):
    problems = datasets.load_dataset(
        "PootieT/MultiPL-E-C2C", f"{args.root_dataset}-{args.lang}",
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
        problems = datasets.Dataset.from_list(problems_list[start_index:stop_index])
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
        "--dataset", type=str, required="--use-local" in sys.argv, help="The local dataset in JSON format to get from this computer."
    )
    # Only required when use local is not passed
    args.add_argument(
        "--src-lang", type=str, required="--use-local" not in sys.argv, default="py", help="Source language for completions"
    )
    args.add_argument(
        "--lang", type=str, required="--use-local" not in sys.argv, help="Target language for completions"
    )
    args.add_argument(
        "--root-dataset", type=str, required="--use-local" not in sys.argv, help="either mbpp or humaneval"
    )
    args.add_argument("--shots", type=int, required="--use-local" not in sys.argv, help="number of shots")
    args.add_argument("--exp", type=str, required="--use-local" not in sys.argv, default="direct",
                      help="type of explanation to use, options={direct, exp, exp-lbl, exp-lbl-d}")
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
        # TODO this needs to be consistent with local dataset output
        args.output_dir = (
            f"{args.src_lang}-{args.lang}/{args.root_dataset}-{args.src_lang}-{args.lang}-{args.exp}-"
            f"{args.shots}shot-{args.temperature}-completion"
        ) if not args.use_local else (
            f"{args.dataset.split('/')[-1].split('.')[0]}-{args.temperature}-completion"
        )

    if args.output_dir_prefix is not None:
        args.output_dir = f"{args.output_dir_prefix}/{args.output_dir}"
    else:
        args.output_dir = f"{model.name}/{args.output_dir}"

    exp_dir = Path(args.output_dir)
    if not exp_dir.exists():
        exp_dir.mkdir()

    if args.use_local:
        problems = from_local_dataset(args)
    else:
        # TODO verify loading from remote huggingface
        problems = from_remote_dataset(args)

    for problem in tqdm(problems, unit="problems"):
        # NOTE(arjun): This is a litte hack to delay loading the model, so that we fail faster.
        problem_filename = exp_dir / f"{problem['name']}.json.gz"
        if problem_filename.exists():
            with gzip.open(problem_filename, "rt") as f:
                existing = json.loads(f.read())
            completions = existing["completions"]
        else:
            completions = []

        if len(completions) > args.completion_limit:
            # Not strictly necessary, but avoid a pointless rewriting of the file with no changes.
            continue

        for _ in tqdm(
            range(len(completions), args.completion_limit, args.batch_size),
            unit="completions",
        ):
            new_completions = model.completions(
                prompt=problem["prompt"],
                max_tokens=750,  #512 normally if no explanation
                temperature=args.temperature,
                n=args.batch_size,
                top_p=0.95,
                stop=problem["stop_tokens"],
            )
            completions.extend(new_completions)

        result_json = {
            "name": problem["name"],
            "language": problem["language"],
            "prompt": problem["prompt"],
            "tests": problem["tests"],
            "completions": completions,
            "stop_tokens": problem["stop_tokens"],
        }
        with gzip.open(problem_filename, "wt") as f:
            json.dump(result_json, f)


if __name__ == "__main__":
    main()
