# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#

import argparse
import json
from logging import getLogger
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import fastBPE
import numpy as np
import pandas as pd
# from spacy.training.gold_io import json_iterate

from codegen_sources.preprocessing.utils import bool_flag
from utils import get_beam_size, add_root_to_path
import pdb
# from pdb import set_trace as breakpoint

add_root_to_path()
from codegen_sources.model.preprocess import XLM_preprocess
from codegen_sources.test_generation.subset_testcases_and_df import read_json, write_json
from sklearn.model_selection import train_test_split
import os

N_SPLITS=4

SOURCE_LANG = "java"

FAILURE = "failure"

TEST_DICT_FIELDS = ["TARGET_CLASS", "tests_strings", "MutationScore",
                    "java_function", "filtered_translation"]  # "first_success_translation", "python_translated_tests"

logger = getLogger()

PATH_TO_TARGET_CLASS_IDS_DEFAULT="/home/shypula/CodeGenMirror/data/target_class_test_ids.txt"
TEST_SET_PROPORTION=0.09562  # about 10K programs, originally 0.05, for QE only
# TEST_SET_PROPORTION = 0.0

TRANSCODER_ST_SUCCESS_CUTOFF = {"java-cpp": 27875, "java-python": 33496, "cpp-python": 14935}
# PATH_TO_SUCCESS_SUBSET_IDS_DEFAULT="/Users/zilutang/Projects/code_translation/CodeGenMirror/data/transcoder_outputs_raw/success_ids.txt"


def get_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input_df", help="path to the input files",
    )
    parser.add_argument(
        "--output_folder", type=str, help="where the files should be outputed",
    )
    parser.add_argument(
        "--langs",
        type=list,
        nargs="+",
        help="List of target langs",
        default=["python", "cpp"],
    )
    parser.add_argument(
        "--bpe_path", type=str, help="where the files should be outputed",
    )
    parser.add_argument(
        "--bpe_vocab", type=str, help="where the files should be outputed",
    )

    parser.add_argument(
        "--path_to_target_class_ids", type=str, help="path to the set of target class ids used for the test set",
    )
    parser.add_argument(
        "--path_to_success_subset_ids", type=str, help="path to the set of success class ids used for the model to "
                                                       "simulate transcoder-ST training scenario",
    )
    parser.add_argument(
        "--program_filter",
        type=str,
        default="success",
        help="filter command to filter dataset with, default selects for all successful programs, one of: "
             "{success, no}",
    )
    parser.add_argument(
        "--output_label",
        type=bool_flag,
        default=False,
        help="True if you want to output label files (for QE model training), false for all other regular data format.",
    )
    parser.add_argument(
        "--cutoff_excess",
        type=bool_flag,
        default=False,
        help="True if you want to mimic transcoder-ST results by randomly select subsets of successful programs, "
             "false to keep all programs that passes unit tests as parallel / QE corpus",
    )
    parser.add_argument(
        "--counter_factual",
        type=bool_flag,
        default=False,
        help="True if you want to add counter-factual examples in QE data: successful but shuffled translations",
    )
    parser.add_argument(
        "--failure_ratio",
        type=float,
        default=-1,
        help="if > 0, down-sample the failure QE data pairs so the final failure-success ratio is this",
    )

    args = parser.parse_args()
    return args


def main(input_path, bpe_model, args):
    langs = ["python"] if len(args.langs) == 1 else args.langs
    input_path = Path(input_path)
    input_dfs_paths = {
        lang: input_path.joinpath(f"test_results_{lang}_df.csv") for lang in langs
    }
    test_results_dfs = {
        lang: pd.read_csv(path) for lang, path in input_dfs_paths.items()
    }
    test_results_dfs = select_tests_several_asserts(test_results_dfs)
    for ref_l in langs[1:]:
        assert len(test_results_dfs[ref_l]) == len(
            test_results_dfs[langs[0]]
        ), f"length of input {len(test_results_dfs[ref_l])} for {ref_l} while it is {len(test_results_dfs[langs[0]])} for {langs[0]}"
        assert (
            test_results_dfs[ref_l][f"{SOURCE_LANG}_function"]
            == test_results_dfs[langs[0]][f"{SOURCE_LANG}_function"]
        ).all(), f"Dataframes order for {ref_l} and {langs[0]} do not match"

    langs = sorted(langs)
    assert len(langs) == len(set(langs)), langs

    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    number_examples = len(test_results_dfs[langs[0]])
    for ref_l in langs[1:]:
        assert number_examples == len(
            test_results_dfs[langs[0]]
        ), f"length of input {number_examples} for {ref_l} while it is {len(test_results_dfs[langs[0]])} for {langs[0]}"
        assert (
            test_results_dfs[ref_l][f"{SOURCE_LANG}_function"]
            == test_results_dfs[langs[0]][f"{SOURCE_LANG}_function"]
        ).all(), f"Dataframes order for {ref_l} and {langs[0]} do not match"

    langs = sorted(langs)
    assert len(langs) == len(set(langs)), langs

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    filtered_code, filtered_label, all_beams_code, all_beams_label = {}, {}, {}, {}
    for lang in langs:
        filtered_code_i, success_label_i, all_code_i, all_label_i = filter_code(test_results_dfs[lang], lang, args.program_filter, args.cutoff_excess)
        filtered_code[lang], filtered_label[lang] = filtered_code_i, success_label_i
        all_beams_code[lang], all_beams_label[lang] = all_code_i, all_label_i
        test_results_dfs[lang]["labels"] = success_label_i

    # filtered_code = {
    #     lang: get_first_success(test_results_dfs[lang], lang) for lang in langs
    # }

    for lang in langs:
        beam_size = get_beam_size(test_results_dfs[lang], results_columns=f"translated_{lang}_functions_beam_")
        # pdb.set_trace()
        filtered_translations_df = test_results_dfs[lang][
            pd.Series(filtered_code[lang]).apply(lambda x: x != FAILURE)
        ]
        filtered_translations_df["filtered_translation"] = [
            c for c in filtered_code[lang] if c != FAILURE
        ]
        filtered_translations_df = filtered_translations_df.drop_duplicates("TARGET_CLASS").reset_index()

        logger.info(
            f"{SOURCE_LANG}-{lang}: {len(filtered_translations_df)} among {number_examples} ({len(filtered_translations_df) / number_examples:.1%})"
        )
        print(
            f"{SOURCE_LANG}-{lang}: {len(filtered_translations_df)} among {number_examples} ({len(filtered_translations_df) / number_examples:.1%})"
        )
        print(
            f"{SOURCE_LANG}-{lang} success label: {sum(filtered_translations_df.labels)} among {number_examples} ({sum(filtered_translations_df.labels) / number_examples:.1%})"
        )

        # save_translations_df(filtered_translations_df, output_folder, lang, split="all")

        if TEST_SET_PROPORTION > 0.00:
            filtered_translations_df_train, filtered_translations_df_test, translated_tests_dicts = split_train_test(
                filtered_translations_df,
                TEST_SET_PROPORTION,
            )

            save_translated_tests(translated_tests_dicts, output_folder, lang)
            # save_translations_df(filtered_translations_df_train, output_folder, lang, split="train")
            # save_translations_df(filtered_translations_df_test, output_folder, lang, split="test")
            java_functions = [r["TARGET_CLASS"] + " | " + r["java_function"] for _, r in
                              filtered_translations_df_test.iterrows()]
            tgt_functions = [r["TARGET_CLASS"] + " | " + r["filtered_translation"] for _, r in
                             filtered_translations_df_test.iterrows()]
            write_bpe_files(
                output_folder,
                java_functions,
                tgt_functions,
                SOURCE_LANG,
                lang,
                bpe_model=bpe_model,
                split="test_st",
                labels=filtered_translations_df_test.labels if args.output_label else None
            )
            filtered_translations_df = filtered_translations_df_train

        java_functions, tgt_functions, labels = extract_java_and_tgt_functions(
            args, beam_size, all_beams_code, all_beams_label, filtered_translations_df, lang)
        write_bpe_files(
            output_folder,
            java_functions,
            tgt_functions,
            SOURCE_LANG,
            lang,
            bpe_model=bpe_model,
            labels=labels if args.output_label else None
        )

    if not args.output_label:
        for lang1 in langs:
            for lang2 in langs[langs.index(lang1) + 1 :]:
                # the only parallel data we have is when the tests are successful for both languages
                filtered_pairs = [
                    (c1, c2)
                    for i, (c1, c2) in enumerate(zip(
                        filtered_code[lang1], filtered_code[lang2]
                    ))
                    if c1 != FAILURE and c2 != FAILURE
                ]
                success_label_pair = [
                    1 if l1 == l2 == 1 else 0
                    for l1, l2, c1, c2 in zip(
                        test_results_dfs[lang1].labels, test_results_dfs[lang2].labels,
                        filtered_code[lang1], filtered_code[lang2]
                    )
                    if c1 != FAILURE and c2 != FAILURE
                ]
                print(
                    f"{lang1}-{lang2}: {sum(success_label_pair)} successsful examples among"
                    f" {number_examples} ({sum(success_label_pair) / number_examples:.1%})"
                    )
                print(
                    f"{lang1}-{lang2}: {len(filtered_pairs)} among {number_examples} ({len(filtered_pairs) / number_examples:.1%})"
                )

                # src_functions, tgt_functions, labels = extract_src_and_tgt_functions(
                #     args, beam_size, filtered_pairs, success_label_pair, filtered_translations_df)

                write_bpe_files(
                    output_folder,
                    [c1 for c1,c2 in filtered_pairs],
                    [c2 for c1,c2 in filtered_pairs],
                    lang1,
                    lang2,
                    bpe_model=bpe_model,
                    labels=success_label_pair if args.output_label else None
                )
    for file_path in Path(output_folder).glob("*.bpe"):
        if len(open(file_path).readlines())<5: 
            continue
        XLM_preprocess(
            str(args.bpe_vocab), str(file_path), str(file_path).replace(".bpe", ".pth")
        )


def extract_src_and_tgt_functions(args, beam_size, filtered_code_pair, filtered_label, filtered_translations_df, test_split=False):
    # in this case, filtered translation df is only useful for its id and target class
    src_functions, tgt_functions, labels = [], [], []

    for i, row in filtered_translations_df.iterrows():
        if args.program_filter == "all_beams" and not test_split:
            for b in range(beam_size):
                src_functions.append(filtered_code_pair[i * beam_size + b][0])
                tgt_functions.append(filtered_code_pair[i * beam_size + b][1])
                labels.append(filtered_label[0][i*beam_size + b])
            post_process_all_beams_functions(src_functions, tgt_functions, labels, args)
        else:
            src_functions.append(row.TARGET_CLASS + " | " + filtered_code_pair[i][0])
            tgt_functions.append(row.TARGET_CLASS + " | " + filtered_code_pair[i][1])
            labels.append(filtered_label[i])

    return src_functions, tgt_functions, labels


def extract_java_and_tgt_functions(args, beam_size, filtered_code, filtered_label, filtered_translations_df, lang):
    if args.program_filter == "all_beams":
        java_functions = [r["java_function"] for _, r in filtered_translations_df.iterrows() for _ in range(beam_size)]
        tgt_functions, labels = [], []
        for i, row in filtered_translations_df.iterrows():
            for b in range(beam_size):
                tgt_functions.append(filtered_code[lang][i * beam_size + b])
                labels.append(filtered_label[lang][i * beam_size + b])
        post_process_all_beams_functions(java_functions, tgt_functions, labels, args)
    else:
        java_functions = filtered_translations_df.java_function
        tgt_functions = filtered_translations_df.filtered_translation
        labels = filtered_translations_df.labels
    return java_functions, tgt_functions, labels


def post_process_all_beams_functions(java_functions, tgt_functions, labels, args):
    # TODO remove duplicate, or maybe subsample failure cases / oversample success cases
    # TODO maybe swap success target to different target class as negative augmentations (swap signatures as well)
    # TODO deobfuscate names / variable names as positive augmentations
    print("post processing all beams function...")
    duplicated_index = pd.Series(tgt_functions).duplicated()
    java_functions = np.array(java_functions)[~duplicated_index]
    labels = np.array(labels)[~duplicated_index]
    tgt_functions = np.array(tgt_functions)[~duplicated_index]
    print(f"removed {sum(duplicated_index)} duplicated index")

    # swap successful programs
    if args.counter_factual:
        suc_mask = labels == 1
        suc_java = java_functions[suc_mask]
        suc_tgt = tgt_functions[suc_mask]
        np.random.shuffle(suc_tgt)
        new_labels = np.array([0] * len(suc_java))
        java_functions = np.concatenate([java_functions, suc_java])
        tgt_functions = np.concatenate([tgt_functions, suc_tgt])
        labels = np.concatenate([labels, new_labels])
        print(f"added {len(new_labels)} new counter-factual examples")

    # downsample failure ones
    if args.failure_ratio > 0:
        fail_idx = labels == 0
        sub_fail_idx = np.random.choice(np.arange(len(labels))[fail_idx],
                                        size=int(args.failure_ratio*sum(labels == 1)), replace=False)
        sub_idx = np.concatenate([np.arange(len(labels))[labels == 1], sub_fail_idx])
        java_functions = java_functions[sub_idx]
        tgt_functions = tgt_functions[sub_idx]
        labels = labels[sub_idx]
        print(f"down-sampled failure cases={sum(labels==0)} (vs. success cases={sum(labels==1)})")
    return java_functions, tgt_functions, labels


def split_train_test(successful_translations_df, test_set_proportion, labels=None):
    """
    Splits the dataframe into train and test sets.
    """
    if os.path.exists(PATH_TO_TARGET_CLASS_IDS):
        target_class_ids = set([id for id in open(PATH_TO_TARGET_CLASS_IDS).read().split("\n") if id])
        print(f"read target class ids from file {PATH_TO_TARGET_CLASS_IDS}, total of {len(target_class_ids)} instances")
    else:
        _ , _test_df = train_test_split(successful_translations_df, test_size=test_set_proportion, random_state=42)
        target_class_ids = set(_test_df["TARGET_CLASS"])
        with open(PATH_TO_TARGET_CLASS_IDS, "w") as f:
            f.write("\n".join(target_class_ids))
        print("wrote {} target class ids to file {}".format(len(target_class_ids), PATH_TO_TARGET_CLASS_IDS))
    train_df = successful_translations_df[~successful_translations_df["TARGET_CLASS"].isin(target_class_ids)]
    test_df = successful_translations_df[successful_translations_df["TARGET_CLASS"].isin(target_class_ids)]
    if not (all(f in test_df.columns for f in TEST_DICT_FIELDS)):
        print("test_df does not have all the fields {}".format(TEST_DICT_FIELDS))
        print("test_df.columns are", test_df.columns)
        print("missing columns are {}".format(set(TEST_DICT_FIELDS).difference(test_df.columns)))
    assert all(f in test_df.columns for f in TEST_DICT_FIELDS)
    test_dicts = [{f: row[f] for f in TEST_DICT_FIELDS} for _, row in test_df.iterrows()]

    return train_df, test_df, test_dicts


def save_translated_tests(translated_tests_dicts, output_folder, lang):
    """
    Saves the translated tests to a file.
    """
    with open (os.path.join(output_folder, f"{lang}_translated_tests.json"), "w") as fh:
        write_json(translated_tests_dicts, fh)


def save_translations_df(successful_translations_df, output_folder, lang, split="train"):
    """
    Saves the dataframe to a file.
    """
    successful_translations_df.to_csv(
        os.path.join(output_folder, f"{lang}_successful_translations_{split}.csv"), index=False
    )


def select_tests_several_asserts(test_results_dfs):
    tests_several_asserts = test_results_dfs["python"].python_translated_tests.apply(
        lambda x: x.count("assert ") > 1
    )

    test_results_dfs = {
        lang: df[tests_several_asserts].reset_index(drop=True)
        for lang, df in test_results_dfs.items()
    }
    new_length = len(test_results_dfs["python"])
    logger.info(
        f"removed {len(tests_several_asserts) - new_length} tests with only one assert ({1 - new_length / len(tests_several_asserts):.2%})"
    )
    return test_results_dfs


def filter_code(test_results, language:str, filter_str:str, cutoff_excess:bool=False):
    if filter_str == "success":
        success_code, label = get_first_success(test_results, language)
    elif filter_str in ["no", "all_beams"]:
        success_code, label = get_first_success(test_results, language, return_first_failed=True)
    else:
        raise NotImplementedError(f"{filter_str} filter is not implemented")

    if filter_str == "all_beams":
        all_success_code, all_label = get_all_beam_results(test_results, language)
    else:
        all_success_code, all_label = None, None

    extra_fail_cnt = sum(label) - TRANSCODER_ST_SUCCESS_CUTOFF[f"{SOURCE_LANG}-{language}"]
    if cutoff_excess and extra_fail_cnt > 0:
        # subset_path = PATH_TO_SUCCESS_SUBSET_IDS_DEFAULT.replace("success_ids", f"{language}_success_ids")
        subset_path = Path(args.path_to_target_class_ids).parent.joinpath(f"{language}_success_ids.txt")
        if os.path.exists(subset_path):
            print("load existing cutoff excess subset")
            # if subset is already sampled, get subset id, select the ones that are
            # not in the subset that is successful, and remove them
            success_subset_ids = [i.strip() for i in open(subset_path).readlines()]
            extra_fail_indices = test_results[(~test_results.TARGET_CLASS.isin(success_subset_ids)) &
                                              (np.array(label).astype(bool))].index
            assert len(extra_fail_indices) == extra_fail_cnt
        else:
            print("cutoff subset not found, run set_random_success_subset first")
            exit()
            np.random.seed(42)
            extra_fail_indices = np.random.choice(np.where(np.array(label)==1)[0], size=extra_fail_cnt, replace=False)

        for idx in extra_fail_indices:
            label[idx] = 0
            if filter_str == "success":
                success_code[idx] = FAILURE
            else:
                success_code[idx] = get_first_failure(test_results.iloc[idx], language)
        print(f"{SOURCE_LANG}-{language}: removed {len(extra_fail_indices)} success to fail, "
              f"with total of {sum(label)} success left")

    return success_code, label, all_success_code, all_label


def get_first_failure(test_result, language):
    beam_size = get_beam_size(
        test_result, results_columns=f"translated_{language}_functions_beam_"
    )
    for i in range(beam_size):
        test_col = f"test_results_{language}_{i}"
        trans_col = f"translated_{language}_functions_beam_{i}"
        if test_result[test_col] != "success":
            return test_result[trans_col]

    print("no failure column found, returning a corrupt successful translation")
    success_translation = test_result[f"translated_{language}_functions_beam_0"]
    toks = success_translation.split()
    n = len(toks)
    toks = np.array(toks)[sorted(np.random.choice(range(n), size=int(0.85*n), replace=False))]
    corrupt_translation = " ".join(toks)
    return corrupt_translation


def get_first_success(test_results, language, return_first_failed: bool=False):
    beam_size = get_beam_size(
        test_results, results_columns=f"translated_{language}_functions_beam_"
    )
    test_results_columns = [f"test_results_{language}_{i}" for i in range(beam_size)]
    translations_columns = [
        f"translated_{language}_functions_beam_{beam}" for beam in range(beam_size)
    ]
    for col in test_results_columns:
        test_results[col] = test_results[col].apply(lambda x: eval(x))
    # pdb.set_trace()
    translations = np.array(
        [test_results[col] for col in translations_columns]
    ).transpose()
    logger.info("getting the first successful function")
    tests_results = np.array(
        [test_results[col] for col in test_results_columns]
    ).transpose()     # ).transpose((1, 0, 2))
    code = []
    labels = []
    min_successful_len = float("inf")
    for translations_i, result_i in zip(translations, tests_results):
        any_successful = False
        # translations_i = sorted(translations_i, key=lambda x: len(x))
        for translated_code, res in zip(translations_i, result_i):
            # if translated_code == "def test ( testval , target ) : NEW_LINE INDENT if testval > target : return + 1 NEW_LINE elif testval < target : return - 1 NEW_LINE else : return NEW_LINE DEDENT":
            #     pdb.set_trace()
            if res[0] == "success":
                if not any_successful:
                    code.append(translated_code)
                    min_successful_len = len(translated_code)
                    any_successful = True
                    labels.append(1)
                elif len(translated_code) < min_successful_len:
                    min_successful_len = len(translated_code)
                    code[-1] = translated_code
        if not any_successful:
            labels.append(0)
            if return_first_failed:
                code.append(translations_i[0])
            else:
                code.append(FAILURE)
    assert len(code) == len(test_results)
    first_successful_code = code
    return first_successful_code, labels


def get_all_beam_results(test_results, language):
    beam_size = get_beam_size(
        test_results, results_columns=f"translated_{language}_functions_beam_"
    )
    test_results_columns = [f"test_results_{language}_{i}" for i in range(beam_size)]
    translations_columns = [
        f"translated_{language}_functions_beam_{beam}" for beam in range(beam_size)
    ]
    success_columns = [f"success_{i}" for i in range(beam_size)]
    for res_col, suc_col in zip(test_results_columns, success_columns):
        test_results[suc_col] = test_results[res_col].apply(lambda x: 1 if x[0]=="success" else 0)
    flatten_code = test_results[translations_columns].to_numpy().flatten()
    flatten_label = test_results[success_columns].to_numpy().flatten()
    return flatten_code, flatten_label


def write_bpe_files(output_folder, lang1_funcs, lang2_funcs, lang1, lang2, bpe_model, split = "train", labels=None):
    if not lang1 < lang2 and labels is None:
        lang1, lang2 = lang2, lang1
        lang1_funcs, lang2_funcs = lang2_funcs, lang1_funcs
    lang1_funcs = bpe_model.apply([f.strip() for f in lang1_funcs])
    lang2_funcs = bpe_model.apply([f.strip() for f in lang2_funcs])
    if labels is None:
        output_files = {
            lang1: [
                open(
                    output_folder.joinpath(
                        f"{split}.{lang1}_sa-{lang2}_sa.{lang1}_sa.{i}.bpe"
                    ),
                    "w",
                )
                for i in range(N_SPLITS)
            ],
            lang2: [
                open(
                    output_folder.joinpath(
                        f"{split}.{lang1}_sa-{lang2}_sa.{lang2}_sa.{i}.bpe" if labels is None else
                        f"{split}.{lang1}_sa-{lang2}_sa.qe_label.{i}.bpe"
                    ),
                    "w",
                )
                for i in range(N_SPLITS)
            ],
        }
    output_files_all = {
        lang1: open(
            output_folder.joinpath(f"{split}.{lang1}_sa-{lang2}_sa.{lang1}_sa.bpe"), "w"
        ),
        lang2: open(
            output_folder.joinpath(f"{split}.{lang1}_sa-{lang2}_sa.{lang2}_sa.bpe" if labels is None else
                                   f"{split}.{lang1}_sa-{lang2}_sa.qe_label.bpe"), "w"
        ),
    }
    if labels is not None:
        assert len(labels) == len(lang1_funcs) == len(lang2_funcs)
        # if label is not None (QE dataset), first dataset output is lang1+lang2, and second is success label
        # <sep> token in transcoder is |, but using <special0> so it can be a unique separator
        if split in ["test_st", "wild", "test_qe", "valid_qe"]:
            # not sure but Alex append and BPE tokenize class name with rest of code for test_st set, I think for
            # unit test self training reasons
            lang1_funcs = [f"{f1} <special0> {f2[f2.find('|') + 2:]}" for f1, f2 in zip(lang1_funcs, lang2_funcs)]
        else:
            lang1_funcs = [f"{f1} <special0> {f2}" for f1, f2 in zip(lang1_funcs, lang2_funcs)]
        lang2_funcs = [str(l) for l in labels]

    for i, (c1, c2) in enumerate(zip(lang1_funcs, lang2_funcs)):
        c1 = c1.strip()
        c2 = c2.strip()
        output_files_all[lang1].write(c1)
        output_files_all[lang1].write("\n")

        output_files_all[lang2].write(c2)
        output_files_all[lang2].write("\n")

        if labels is None:
            output_files[lang1][i % N_SPLITS].write(c1)
            output_files[lang1][i % N_SPLITS].write("\n")

            output_files[lang2][i % N_SPLITS].write(c2)
            output_files[lang2][i % N_SPLITS].write("\n")

    if labels is None:
        for o in output_files[lang1] + output_files[lang2]:
            o.close()
    for o in output_files_all.values():
        o.close()


def write_big_clone_bench_format(output_folder, lang1_funcs: Dict[str, List[str]], lang2_funcs, lang1, lang2, bpe_model, labels):
    if not lang1 < lang2:
        lang1, lang2 = lang2, lang1
        lang1_funcs, lang2_funcs = lang2_funcs, lang1_funcs
    output_data_file = open(output_folder.joinpath("data.jsonl"))
    output_splits_file = {
        splt: open(
            output_folder.joinpath(f"{splt}.txt"), "w"
        ) for splt in lang1_funcs.keys()
    }

    for splt in lang1_funcs.keys():
        for i, (c1, c2) in enumerate(zip(lang1_funcs[splt], lang2_funcs[splt])):
            c1 = c1.strip()
            c2 = c2.strip()
            output_data_file.write(json.dumps({"func": c1, "idx": i * 2}) + "\n")
            output_data_file.write(json.dumps({"func": c2, "idx": i * 2+1}) + "\n")

            output_splits_file[splt].write(f"{i*2} {i*2+1} {labels[splt][i]}")

    for o in list(output_splits_file.values()) + [output_data_file]:
        o.close()


def set_random_success_subset(in_folder):
    cpp_df = pd.read_csv(f"{in_folder}/test_results_cpp_df.csv")
    py_df = pd.read_csv(f"{in_folder}/test_results_python_df.csv")
    dfs = select_tests_several_asserts({"cpp": cpp_df, "python": py_df})
    cpp_df, py_df = dfs["cpp"], dfs["python"]
    _, cpp_label = get_first_success(cpp_df, "cpp")
    _, py_label = get_first_success(py_df, "python")
    cpp_success_ids = cpp_df.TARGET_CLASS[np.array(cpp_label).astype(bool)]
    py_success_ids = py_df.TARGET_CLASS[np.array(py_label).astype(bool)]

    # randomly subset until one has closest intersect count (cpp-python) as original
    min_diff = 1000000
    # min_seed = 0
    # min_cpp_ids, min_py_ids = None, None
    # for seed in tqdm(range(30)):
    #     np.random.seed(seed)
    #     cpp_subset_ids = np.random.choice(cpp_success_ids, TRANSCODER_ST_SUCCESS_CUTOFF["java-cpp"], replace=False)
    #     py_subset_ids = np.random.choice(py_success_ids, TRANSCODER_ST_SUCCESS_CUTOFF["java-python"], replace=False)
    #     intersect_ids = set(cpp_subset_ids).intersection(py_subset_ids)
    #     diff = np.abs(TRANSCODER_ST_SUCCESS_CUTOFF['cpp-python']-len(intersect_ids))
    #     print(f"seed={seed}, cpp-python size={len(intersect_ids)}, diff={diff}")
    #     if diff < min_diff:
    #         min_diff, min_seed = diff, seed
    #         min_cpp_ids, min_py_ids = cpp_subset_ids, py_subset_ids
    # print(f"best seed={min_seed}, best diff={min_diff}")

    # subset cpp-python translations, then subset the rest of cpp / python to make up for the difference
    intersect_ids = set(cpp_success_ids).intersection(py_success_ids)
    np.random.seed(42)
    intersect_subset_ids = set(np.random.choice(list(intersect_ids), TRANSCODER_ST_SUCCESS_CUTOFF["cpp-python"], replace=False))
    left_cpp = set(cpp_success_ids).difference(intersect_ids)
    left_cpp_cnt = TRANSCODER_ST_SUCCESS_CUTOFF["java-cpp"] - TRANSCODER_ST_SUCCESS_CUTOFF["cpp-python"]
    cpp_subset_ids = intersect_subset_ids.copy()
    cpp_subset_ids.update(np.random.choice(list(left_cpp), left_cpp_cnt, replace=False))
    left_py = set(py_success_ids).difference(intersect_subset_ids)
    left_py_cnt = TRANSCODER_ST_SUCCESS_CUTOFF["java-python"] - TRANSCODER_ST_SUCCESS_CUTOFF["cpp-python"]
    py_subset_ids = intersect_subset_ids.copy()
    py_subset_ids.update(np.random.choice(list(left_py), left_py_cnt, replace=False))

    assert len(py_subset_ids) == TRANSCODER_ST_SUCCESS_CUTOFF["java-python"]
    assert len(cpp_subset_ids) == TRANSCODER_ST_SUCCESS_CUTOFF["java-cpp"]
    assert len(py_subset_ids.intersection(cpp_subset_ids)) == TRANSCODER_ST_SUCCESS_CUTOFF["cpp-python"]

    with open(PATH_TO_SUCCESS_SUBSET_IDS_DEFAULT.replace("success_id", "cpp_success_id"), "w") as f:
        f.writelines([i+"\n" for i in cpp_subset_ids])
    with open(PATH_TO_SUCCESS_SUBSET_IDS_DEFAULT.replace("success_id", "python_success_id"), "w") as f:
        f.writelines([i+"\n" for i in py_subset_ids])


if __name__ == "__main__":
    args = get_arguments()
    # local debug
    # root = "/Users/zilutang/Projects/code_translation/CodeGenMirror/data/transcoder_outputs_raw"
    # args.input_df = f"{root}/test_results"
    # args.output_folder =f"{root}/offline_dataset_qe_all_beams"
    args.langs = ["python", "cpp"]  # cpp
    # args.bpe_path = "/Users/zilutang/Projects/code_translation/CodeGenMirror/data/bpe/cpp-java-python/codes"
    # args.bpe_vocab = "/Users/zilutang/Projects/code_translation/CodeGenMirror/data/bpe/cpp-java-python/vocab"
    # args.path_to_target_class_ids = f"{root}/test_set_ids.txt"
    # args.program_filter = "all_beams"
    # args.output_label = True
    # args.cutoff_excess = True
    # args.counter_factual=True
    # args.failure_ratio=2

    bpe_model = fastBPE.fastBPE(args.bpe_path)
    global path_to_target_class_ids
    PATH_TO_TARGET_CLASS_IDS = args.path_to_target_class_ids if args.path_to_target_class_ids else PATH_TO_TARGET_CLASS_IDS_DEFAULT
    main(
        Path(args.input_df),
        bpe_model,
        args
    )
    # file_path = "/Users/zilutang/Projects/code_translation/CodeGenMirror/data/transcoder_outputs_raw/offline_dataset_qe_all_beams/train.java_sa-qe_label.qe_label.bpe"
    # XLM_preprocess(
    #     str(args.bpe_vocab), str(file_path), str(file_path).replace(".bpe", ".pth")
    # )
    # set_random_success_subset("/Users/zilutang/Projects/code_translation/CodeGenMirror/data/transcoder_outputs_raw/test_results")
"""
python codegen_sources/test_generation/select_successful_tests.py \
    --input_df /nobackup/users/shypula/ST_output_reproduction_1_14/results/test_results_new \
    --output_folder /nobackup/users/shypula/ST_output_reproduction_1_14/offline_dataset_qe_all_beams_dedup_cf \
    --bpe_path ./data/bpe/cpp-java-python/codes \
    --bpe_vocab ./data/bpe/cpp-java-python/vocab \
    --path_to_target_class_ids ./data/transcoder_outputs_raw/test_set_ids.txt \
    --program_filter all_beams \
    --output_label true \
    --cutoff_excess true \
    --counter_factual true \
    --failure_ratio -1 
"""