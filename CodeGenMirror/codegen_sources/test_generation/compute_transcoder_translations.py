# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os.path
import traceback
from itertools import repeat
from logging import getLogger
from pathlib import Path

import pandas as pd
from submitit import AutoExecutor, LocalExecutor
from tqdm import tqdm
from utils import chunks_df, add_root_to_path
import pdb

add_root_to_path()
from codegen_sources.model.src.utils import set_MKL_env_vars
from codegen_sources.model.translate import Translator
from codegen_sources.preprocessing.utils import bool_flag
from codegen_sources.test_generation.compute_test_results import compute_test_results
import torch.multiprocessing as mp
import sys

# from codegen_sources.test_generation.reformat_df_java_only import JAVA_COL, parallel_transform

CHUNKSIZE = 2500
SUPPORTED_LANGUAGES = ["python", "cpp"]
primitive_types = {"short", "int", "long", "float", "double", "boolean", "char"}
logger = getLogger()

java_standard_types = {
    "Double",
    "Float",
    "String",
    "Integer",
    "Boolean",
    "Long",
    "Short",
}
java_simple_types = primitive_types | java_standard_types
java_supported_types = (
    java_simple_types
    | {f"{t}[]" for t in java_simple_types}
    | {f"ArrayList<{t}>" for t in java_simple_types}
)


def get_joined_func_tests_df(csv_path, functions_path):
    assert Path(csv_path).is_file(), csv_path
    tests_dataframe = pd.read_csv(csv_path)
    java_functions_path = Path(functions_path)

    # reading functions to DF
    java_functions = [
        func
        for f in java_functions_path.glob("java.0000*.sa.tok")
        for func in open(f).readlines()
    ]
    java_functions = pd.DataFrame(
        {
            "func_ids": [f.split(" | ")[0] for f in java_functions],
            "java_function": [f.split(" | ")[1] for f in java_functions],
        }
    )

    # getting the IDs of the functions. The class name is created from it
    tests_dataframe["func_ids"] = tests_dataframe.TARGET_CLASS.apply(
        lambda x: x.replace("CLASS_", "", 1)
    )
    merged = tests_dataframe.merge(java_functions, how="inner", on="func_ids")
    return merged


def compute_transcoder_translation(
    df, output_file, model_path, bpe_path, target_language, beam_size=20, device_no=None, obfuscated=False
):
    transcoder = Translator(model_path, bpe_path, device_no)
    res = [[] for _ in range(beam_size)]
    suffux_1 = suffix_2 = "_obfuscated" if obfuscated else "_sa"
    for i, func in enumerate(df["java_function"]):
        if i % 100 == 0:
            logger.info(f"computed {i} translations / {len(df)}")
        try:
            translations = transcoder.translate_with_transcoder(
                func,
                "java",
                target_language,
                beam_size=beam_size,
                detokenize=False,
                max_tokens=1024,
                length_penalty=0.5,
                suffix1=suffux_1,
                suffix2=suffix_2,
                device = f"cuda:{device_no}" if device_no is not None else "cuda:0",
            )
        except RuntimeError:
            logger.error(f"RuntimeError for {func}")
            traceback.print_exc()
            translations = ["Failure"] * beam_size
        for i, res_i in enumerate(translations):
            res[i].append(res_i)

    for i, res_i in enumerate(res):
        df[f"translated_{target_language}_functions_beam_{i}"] = res_i
    df.to_csv(output_file, index=False)
    
    
def compute_transcoder_translation_multi(df, output_file, model_path, bpe_path, target_language, beam_size=20,
                                         n_gpus=None, obfuscated=False):
    if n_gpus <= 1:
        return compute_transcoder_translation(df, output_file, model_path, bpe_path, target_language, beam_size, obfuscated)
    else:
        jobs = []
        len_df = len(df) // (n_gpus*2)
        if len(df) > 50:
            dfs = list(chunks_df(df, len_df))
            if len(dfs) > (n_gpus*2):
                dfs = dfs[:-2] + [pd.concat(dfs[-2:])]
        else:
            dfs = [df]
        out_files = []
        for i in range(len(dfs)):
            out_f = os.path.splitext(output_file)
            out_f = out_f[0] + f"_{i}.csv"
            print("jobs has df with len {} and output file {} and device {}".format(len(dfs[i]), out_f, (i%n_gpus)),
                  flush=True, file=sys.stderr)
            jobs.append(mp.Process(target=compute_transcoder_translation, 
                                   args=(dfs[i], out_f, model_path, bpe_path, target_language, beam_size, (i%n_gpus), obfuscated)))
            out_files.append(out_f)
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        base_dir = os.path.dirname(output_file)
        if any([not os.path.exists(f) for f in out_files]):
            print("Some files were not created", flush=True, file=sys.stderr)
            print("Outfiles: " + "\n".join(out_files), flush=True, file=sys.stderr)
            print("Missing: " + "\n".join([f for f in out_files if not os.path.exists(f)]), flush=True, file=sys.stderr)
        out_files = [f for f in out_files if os.path.exists(f)]
        df = pd.concat([pd.read_csv(f) for f in out_files], ignore_index=True).drop_duplicates("TARGET_CLASS")
        print("concatenated df with len {} if ignoring index and using the list of outfiles".format(len(df)), flush=True, file=sys.stderr)
        df.to_csv(output_file, index=False)
        
    
def main(args):
    assert args.n_gpus > 0 and isinstance(args.n_gpus, int)
    output_folder = Path(args.output_folder)
    # output_folder.mkdir(exist_ok=True, parents=True)
    transcoder_output_folder = "transcoder_outputs"
    output_folder_translations = output_folder.joinpath(transcoder_output_folder)
    if args.local is False:
        logger.info("Executing on cluster")
        cluster = AutoExecutor(output_folder_translations.joinpath("log"))
        cluster.update_parameters(
            cpus_per_task=40,
            gpus_per_node=args.n_gpus,
            mem_gb=400,
            timeout_min=1400,
            # constraint="volta32gb",
            slurm_partition="sched_system_all_8",
            slurm_array_parallelism=3,
            exclude="node0041,node0016"
        )
    else:
        logger.info("Executing locally")
        cluster = LocalExecutor(output_folder_translations.joinpath("log"))
        cluster.update_parameters(timeout_min=4319,)
    merged_df = get_joined_func_tests_df(args.csv_path, args.functions_path)
    merged_df = merged_df.drop_duplicates(subset="TARGET_CLASS")
    merged_df = merged_df.drop_duplicates(subset="java_function")
    if args.obfuscate_java_functions:
        merged_df = merged_df.reset_index()
        merged_df["java_function_orig"] = merged_df["java_function"]
        obfuscated_java_functions, n_success = parallel_transform(merged_df["java_function"],
                                                                  "java",
                                                                  add_spans=False,
                                                                  obfuscate=True
                                                                  )
        merged_df["java_function"] = pd.Series(obfuscated_java_functions)
        print(f"Obfuscated {n_success} java functions of {len(merged_df)}")
    print("merged df has {} rows".format(len(merged_df)), flush=True, file=sys.stderr)
    CHUNKSIZE = len(merged_df) // 6
    chunks = list(chunks_df(merged_df, CHUNKSIZE))
    if len(chunks) > 6:
        chunks = chunks[:-2] + [pd.concat(chunks[-2:])]
    merged_chunks = pd.concat(chunks, ignore_index=True)
    print("merged chunks has {} rows".format(len(merged_chunks)), flush=True, file=sys.stderr)
    del merged_chunks
    output_files = [
        output_folder_translations.joinpath(f"{args.target_language}_chunk_{i}.csv")
        for i in range(len(chunks))
    ]
    logger.info(f"{len(chunks)} chunks of size {len(chunks[0])}")
    missing_output_files = output_files
    if not args.rerun:
        indices_to_run = [i for i, p in enumerate(output_files) if not (p.is_file())]
        # indices_to_run = [8]
        logger.info(
            f"Running on the remaining {len(indices_to_run)} among {len(output_files)} files"
        )
        chunks = [chunks[i] for i in indices_to_run]
        missing_output_files = [output_files[i] for i in indices_to_run]
    assert len(chunks) == len(missing_output_files)
    if len(chunks) > 0:
        jobs = cluster.map_array(
            compute_transcoder_translation_multi,
            chunks,
            missing_output_files,
            repeat(args.model_path),
            repeat(args.bpe_path),
            repeat(args.target_language),
            repeat(args.beam_size),
            repeat(args.n_gpus),
            repeat(args.obfuscate_java_functions),
        )
        for j in tqdm(jobs):
            j.result()
    chunks_files = [
        output_folder_translations.joinpath(f"{args.target_language}_chunk_{i}.csv")
        for i in range(len(output_files))
    ]
    output_csv_path = output_folder_translations.joinpath(
        f"{args.target_language}_transcoder_translation.csv"
    )
    df = pd.concat([pd.read_csv(chunk) for chunk in chunks_files], axis=0, ignore_index=True)
    print("the length of the merged final df is {}".format(len(df)), flush=True, file=sys.stderr)
    df.to_csv(output_csv_path, index=False)
    compute_test_results(
        output_csv_path,
        args.target_language,
        output_folder.joinpath("test_results"),
        local=args.local,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--functions_path", help="path to the input files",
    )
    parser.add_argument(
        "--csv_path", help="path to the input test csv",
    )
    parser.add_argument(
        "--output_folder", help="output path",
    )
    parser.add_argument(
        "--target_language", help="target language. python or cpp", default="python",
    )
    parser.add_argument(
        "--model_path", type=str, help="where the files should be outputed",
    )
    parser.add_argument(
        "--bpe_path", type=str, help="where the files should be outputted",
    )
    parser.add_argument(
        "--local",
        type=bool_flag,
        default=True,
        help="True if you want to run the processing pipeline locally, false if want to use submitit.",
    )
    parser.add_argument(
        "--rerun",
        type=bool_flag,
        default=False,
        help="True if you want to run the processing pipeline locally, false if want to use submitit.",
    )
    
    parser.add_argument(
        "--beam_size",
        type=int,
        default=20,
        help="Number of beams to use in beam search",
    )
    
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=1,
        help="Set to a positive integer for the number of GPUs to use if using multiprocessing",
    )

    parser.add_argument(
        "--obfuscate_java_functions",
        type=bool_flag,
        default=False,
        help="True if you want to obfuscate java functions",
    )

    # parser.add_argument('--filter_several_tests', type=bool_flag, default=True, help='Filter to keep only the examples with at least 2 tests')
    args = parser.parse_args()
    assert Path(args.bpe_path).is_file(), args.bpe_path
    assert Path(args.model_path).is_file()
    assert args.target_language in SUPPORTED_LANGUAGES
    return args


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    logger.info("#" * 10 + "Computing Translations" + "#" * 10)
    set_MKL_env_vars()
    args = parse_arguments()
    main(args)
    logger.info("\n" * 2)
