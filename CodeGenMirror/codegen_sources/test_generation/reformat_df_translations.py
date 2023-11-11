import argparse
from logging import getLogger
from pathlib import Path
import fastBPE
import numpy as np
import pandas as pd
import math
import os
import re
import subprocess
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path, PosixPath
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from itertools import repeat

import argparse
from submitit import AutoExecutor, LocalExecutor
from tqdm import tqdm
import pandas as pd
from utils import add_root_to_path
from codegen_sources.preprocessing.utils import bool_flag
from data.ast.JavaSpanAnnotator import JavaSpanAnnotator
from data.ast.PythonSpanAnnotator import PythonSpanAnnotator
from codegen_sources.model.src.evaluation.comp_acc_computation import convert_filled_arguments

import pdb
import traceback

# f = "static void test_ci_neg ( int [ ] a , float [ ] b ) { for ( int i = a . length - 1 ; i >= 0 ; i -= 1 ) { a [ i ] = - 123 ; b [ i ] = - 103.f ; } }"
# convert_filled_arguments(f, f, "java", JavaSpanAnnotator.processor)

JAVA_COL = "java_function"
PY_COL="translated_python_functions_beam_[0-9]+"


def get_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input_df", help="path to the input df",
    )
    parser.add_argument(
        "--output_df", required=False, help="path to the output df",
    )
    parser.add_argument(
        "--spans",
        type=bool_flag,
        default=True,
        help="True if you want to run the processing pipeline locally, false if want to use submitit.",
    )
    parser.add_argument(
        "--obfuscate",
        type=bool_flag,
        default=False,
        help="True if you want to run the processing pipeline locally, false if want to use submitit.",
    )

    args = parser.parse_args()
    return args

def parallel_transform(input_progs, lang, add_spans, obfusate):
    results = []
    pbar = tqdm(total=len(input_progs), desc="parallel_transform for language {}".format(lang))
    n_failed = 0
    if lang == "java":
        pool = Pool(int(cpu_count()*1.5))
        m = pool.map
    else:
        pool = Pool(cpu_count())
        m = pool.map
    args_list = [(input_prog, lang, add_spans, obfusate) for input_prog in input_progs]
    #pdb.set_trace()
    for result, success in m(_transform, args_list):
        results.append(result)
        pbar.update(1)
        if result == "Failed" or success == False: 
            n_failed+=1
            pbar.set_description("n_failed is {}, lang is {}".format(n_failed, lang))
    if any([type(c) != str for c in results]): 
        print("not type str: ", [c for c in results if type(c) != str]) 
        pdb.set_trace()
    series = pd.Series(results)
    if any([type(c) != str for c in series]): 
        pdb.set_trace()
        print("not type str: ", [c for c in series if type(c) != str]) 
    return results

def _transform(args):
    return transform(*args)

def transform(input_prog, lang, add_spans, obfusate):
    #pdb.set_trace()
    assert lang in ("python", "java")
    assert any([add_spans, obfusate])
    success = True
    try:
        annotator_class = JavaSpanAnnotator if lang == "java" else PythonSpanAnnotator
        annotator = annotator_class(input_prog, obfusate)
        if add_spans:
            #annotator = annotator_class(input_prog, obfusate)
            prog_str = annotator.get_annotated_prog_str(tokenized_style=True)
        elif obfusate:
            prog_str = annotator.get_prog(tokenized_style=True) 
        if type(prog_str) != str:
            #pdb.set_trace()
            print("for prog:\n{} encountered error:".format(input_prog))
            print("prog string was returned as None")
            prog_str = input_prog
            #prog_str = annotator_class.obfuscate_code(input_prog)
            success = False
    except Exception as e:
        if lang == "java":
            print("for prog:\n{} encountered error:\n{}".format(input_prog, e))
            prog_str = "Failed"
        else:
            #print("for prog:\n{} encountered error:\n{}".format(input_prog, e))
            #traceback.print_exc()
            prog_str = input_prog
        success = False
    return prog_str, success


def main(args):
    # df = pd.read_csv(os.path.join(args.input_df, "test_results_python_df.csv"))
    df = pd.read_csv(os.path.join(args.input_df))
    df = df.drop_duplicates(subset="TARGET_CLASS").reset_index()
    py_cols = [c for c in df.columns if re.match(PY_COL, c)]
    df[JAVA_COL] = pd.Series(parallel_transform(df[JAVA_COL], "java", args.spans, args.obfuscate))
    n_failed = sum(df[JAVA_COL] == "Failed")
    print(f"{n_failed} out of {len(df)} total rows for java")
    #df = df[df[JAVA_COL] != "Failed"]
    pbar = tqdm(total=len(py_cols))
    for col in py_cols:
        pbar.update(1)
        result = parallel_transform(df[col], "python", args.spans, args.obfuscate)
        if any([type(c) != str for c in result]): 
                pdb.set_trace()
        if any([type(c) != str for c in pd.Series(result)]): 
                pdb.set_trace()
        df[col] = pd.Series(result)
        if any([type(c) != str for c in df[col]]): 
                pdb.set_trace()
    df = df[df[JAVA_COL] != "Failed"]
    output_string = "_".join(["spans" if args.spans else "no_spans", "obfusate" if args.obfuscate else "no_obfusate"])
    if not args.output_df:
        args.output_df = os.path.join(args.input_df, f"test_results_python_df_{output_string}.csv")
    df.to_csv(args.output_df, index=False)
    return 0

if __name__ == "__main__":
    args = get_arguments()
    main(args)
