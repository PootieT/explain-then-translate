# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os.path
import pdb

import fastBPE
import torch
from pathlib import Path, PosixPath
import pandas as pd

from codegen_sources.preprocessing.utils import bool_flag

INITIAL_CACHE_FOLDER = "initial_cache"

LANGUAGES = ["java", "python", "cpp"] # ["cpp",
from logging import getLogger

import numpy as np
from utils import ROOT_PATH, add_root_to_path

add_root_to_path()
from codegen_sources.model.src.cache import ListCache
from codegen_sources.model.preprocess import XLM_preprocess
from codegen_sources.model.src.data.dataset import MUTATION_SCORE, ASSERTS_COUNT

logger = getLogger()


class Params:
    def __init__(self, pad_index=0, eos_index=1):
        self.pad_index = pad_index
        self.eos_index = eos_index
        self.tokens_per_batch = 1000
        self.st_remove_proba = 0.3


def get_tensors(reloaded_data):
    pos = reloaded_data["positions"]
    sent = reloaded_data["sentences"]

    sentences = [sent[p[0] : p[1]] for p in pos]

    lengths = [torch.tensor(len(s) + 2) for s in sentences]

    out_sentences = []
    for s in sentences:
        l = len(s) + 2
        out_s = torch.LongTensor(l).fill_(1)
        out_s[1 : l - 1].copy_(torch.from_numpy(s.astype(np.int64)))
        out_sentences.append(out_s)
    return out_sentences, lengths


def initialize_cache(dataset_path, output_path: PosixPath):
    languages = [l + "_sa" for l in LANGUAGES]
    for l1 in languages:
        for l2 in [l for l in languages if l > l1]:
            print(f"computing initial cache for {l1}-{l2}")
            sents1, len1 = get_tensors(
                torch.load(dataset_path.joinpath(f"train.{l1}-{l2}.{l1}.pth"))
            )
            sents2, len2 = get_tensors(
                torch.load(dataset_path.joinpath(f"train.{l1}-{l2}.{l2}.pth"))
            )

            assert len(sents1) == len(sents2) == len(len1) == len(len2)

            elements = list(zip(sents1, len1, sents2, len2))

            ListCache(elements, Params()).save(
                output_path.joinpath(f"cache_{l1}-{l2}.pkl")
            )


def add_self_trained_dataset(data_df, dataset_path, vocab_path, with_qe=False):
    logger.info(f"Self labelled dataset to {dataset_path}")
    bpe_model = fastBPE.fastBPE(
        str(ROOT_PATH.joinpath("data/bpe/cpp-java-python/codes"))
    )
    print("unfiltered df:", len(data_df))
    if not with_qe:
        data_df = data_df[
            data_df.python_translated_tests.apply(lambda x: x.count("assert")) > 1
        ]
    print("filtered df:", len(data_df))
    java_functions_with_indices = bpe_model.apply(
        pd.Series(data_df["TARGET_CLASS"] + " | " + data_df["java_function"])
    )
    output_folder = dataset_path
    output_files = [
        open(
            output_folder.joinpath(f"self_training.java_sa.{i}.bpe"),
            "w",
            encoding="utf-8",
            errors="ignore",
        )
        for i in range(args.n_gpus)
    ]
    output_files_all = open(
        output_folder.joinpath(f"self_training.java_sa.bpe"),
        "w",
        encoding="utf-8",
        errors="ignore",
    )
    for i, l in enumerate(sorted(java_functions_with_indices)):
        output_files_all.write(l.strip())
        output_files_all.write("\n")

        output_files[i % args.n_gpus].write(l.strip())
        output_files[i % args.n_gpus].write("\n")
    for f in output_files:
        f.close()
    output_files_all.close()
    for file_path in Path(output_folder).glob("*.bpe"):
        print(f"Processing {file_path} with vocab {Path(vocab_path).absolute()}")
        XLM_preprocess(
            str(Path(vocab_path).absolute()),
            str(file_path),
            str(file_path).replace(".bpe", ".pth"),
        )


def output_multilingual_tests_dataset(df_python, df_cpp, output_path):
    data_df = df_python[
        ["TARGET_CLASS", "java_function", "path_to_test", "python_translated_tests"]
    ]
    data_df["cpp_translated_tests"] = df_cpp["cpp_translated_tests"]
    data_df[MUTATION_SCORE] = df_python["MutationScore"]
    data_df[ASSERTS_COUNT] = data_df.python_translated_tests.apply(
        lambda x: x.count("assert")
    )

    data_df[
        [
            "TARGET_CLASS",
            "path_to_test",
            "python_translated_tests",
            "cpp_translated_tests",
            MUTATION_SCORE,
            ASSERTS_COUNT,
        ]
    ].to_json(
        output_path.joinpath("translated_tests.json"), orient="records", lines=True
    )
    return data_df


def load_wild_dataset(test_res_dir: Path):
    # TODO ideally this path should be passed in as argument
    transcoder_output_dir = test_res_dir.parent.joinpath("transcoder_outputs")
    df = pd.DataFrame()
    for lang in ["cpp", "python"]:
        for split in ["wild", "wilder"]:
            split_path = transcoder_output_dir.joinpath(f"{lang}_transcoder_translation_{split}.csv")
            if os.path.exists(split_path):
                split_df = pd.read_csv(split_path)[["TARGET_CLASS", "java_function"]]
                df = pd.concat([df, split_df]).drop_duplicates()
    pdb.set_trace()
    return df


if __name__ == "__main__":
    logger.info("#" * 10 + "Creating data for Online Self-Training" + "#" * 10)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_path", help="path to the offline dataset",
    )

    parser.add_argument(
        "--input_dfs_path", help="Path to input dataframes",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="where the files should be outputed",
        default=Path(ROOT_PATH).joinpath("data"),
    )
    parser.add_argument(
        "--n_gpus", type=int, help="number of train set splits", default=8
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        help="Path to vocab",
        default=Path(ROOT_PATH).joinpath("data", "bpe", "cpp-java-python", "vocab"),
    )
    parser.add_argument(
        "--with_qe",
        type=bool_flag,
        default=False,
        help="Wether the reward signal should come from QE model (true) or from unit tests (false ",
    )

    args = parser.parse_args()
    output_path = Path(args.output_path)
    dataset_path = Path(args.dataset_path)

    # for wilder datasets, just point to wilder dataset folder, as long as we have
    # run at least one model finetuning on that set, the train. file should contain
    # wild data as well
    initialize_cache(dataset_path, output_path.joinpath(INITIAL_CACHE_FOLDER))

    input_dfs_path = Path(args.input_dfs_path)
    assert input_dfs_path.is_dir()
    if not args.with_qe:
        input_dfs_paths = {
            lang: input_dfs_path.joinpath(f"test_results_{lang}_df.csv")
            for lang in ["python", "cpp"] #, "cpp"]
        }
        test_results_dfs = {
            lang: pd.read_csv(path) for lang, path in input_dfs_paths.items()
        }
        data_df = output_multilingual_tests_dataset(
            test_results_dfs["python"], test_results_dfs["cpp"], output_path
        )
    else:
        data_df = load_wild_dataset(input_dfs_path)
    # TODO, to adapt online learning with wilder dataset, add another one here
    add_self_trained_dataset(data_df, output_path, args.vocab_path, args.with_qe)

    logger.info("\n" * 2)

"""
python codegen_sources/test_generation/create_data_for_online_st.py \
    --dataset_path /nobackup/users/shypula/ST_output_reproduction_1_14/offline_dataset_cutoff \
    --input_dfs_path /nobackup/users/shypula/ST_output_reproduction_1_14/results/test_results_new \
    --output_path /nobackup/users/shypula/ST_output_reproduction_1_14/online_ST_files_cutoff \
    --vocab_path /nobackup/users/zilutang/CodeGenMirror/data/bpe/cpp-java-python/vocab \
    --with_qe False \
    --n_gpus 4

python codegen_sources/test_generation/create_data_for_online_st.py \
    --dataset_path /nobackup/users/shypula/ST_output_reproduction_1_14/offline_dataset_wilder_filt_pct_0.7_AB \
    --input_dfs_path /nobackup/users/shypula/ST_output_reproduction_1_14/results/test_results_new \
    --output_path /nobackup/users/shypula/ST_output_reproduction_1_14/online_ST_files_wilder \
    --vocab_path /nobackup/users/zilutang/CodeGenMirror/data/bpe/cpp-java-python/vocab \
    --with_qe true \
    --n_gpus 4
"""