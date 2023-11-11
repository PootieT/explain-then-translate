import argparse
import pdb

import codegen_sources.preprocessing.lang_processors.python_processor
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor

from codegen_sources.test_generation.test_runners.cpp_test_runner import CppTestRunner
from codegen_sources.test_generation.test_runners.python_test_runner import (
    PythonTestRunner,
)

from codegen_sources.model.src.utils import (
    TREE_SITTER_ROOT,
    limit_virtual_memory,
    MAX_VIRTUAL_MEMORY,
    bool_flag
)
from codegen_sources.model.src.evaluation.comp_acc_computation import run_python_program, TOFILL, EXT
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pdb
from pathlib import Path

python_runner = PythonTestRunner()

def get_parser():
    parser = argparse.ArgumentParser(description="Generate tests for a function")
    parser.add_argument("--input_dataframe_path", type=str, required=True)
    parser.add_argument("--workers", type=str, required=False, default="1")
    parser.add_argument("--dump_failed_progs", type=bool_flag, required=False, default=False)
    parser.add_argument("--subset_of_df", type=int, required=False, default=None)
    return parser

def test_function(function, translated_test_string, lang, i, outfolder="tmp_scripts"):
    if lang != "python":
        raise Exception("Only python supported")
    lang_processor =  LangProcessor.processors["python"](root_folder=TREE_SITTER_ROOT)
    result = python_runner.get_tests_results(function, translated_test_string)

    return result, function, translated_test_string


def main():
    args = get_parser().parse_args()
    df = pd.read_csv(args.input_dataframe_path)
    if args.subset_of_df is not None: 
        assert type(args.subset_of_df) == int and args.subset_of_df > 0 and args.subset_of_df < len(df)
        df = df[:args.subset_of_df]
    workers = args.workers
    jobs = [(r["first_successful_translation"], r["python_translated_tests"], "python", i) for i, r in df.iterrows()]
    pbar = tqdm(total=len(jobs))
    successes = 0
    n_processed = 0
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        for i, (r, f, t) in enumerate(executor.map(test_function, *zip(*jobs))):
            pbar.update(1)
            n_processed += 1
            if r[0] == "success":
                successes += 1
            elif args.dump_failed_progs:
                print(f"\n\n{'#'*20}Failed program at idx {i} was {'#'*20}\n\n{f}\n\n")
                print(f"{'#'*20}Failed test string at idx {i} for it was {'#'*20}\n\n{t}\n\n")
                #print(r[1])
            pbar.set_description(f"Successes: {successes}, proportion: {successes / n_processed:.2f}")

if __name__ == "__main__":
    main()




