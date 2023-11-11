import argparse
import itertools
import shutil

from codegen_sources.model.src.utils import bool_flag
from pathlib import Path
from codegen_sources.preprocessing.bpe_modes.fast_bpe_mode import FastBPEMode, BPEMode
from data.ast.PythonSpanAnnotator import PythonSpanAnnotator
from data.ast.JavaSpanAnnotator import JavaSpanAnnotator
import subprocess
from typing import List
import os
import pdb
from tqdm import tqdm
import traceback
from javalang.tokenizer import LexerError 
from javalang.parser import JavaSyntaxError
from codegen_sources.model.preprocess import XLM_preprocess
import pandas as pd

ROOT_DIR_TO_TESTS="online_ST_files/translated_tests.json"
ROOT_DIR_TO_TESTS_TEMP="online_ST_files/translated_tests_temp.json"
ROOT_DIR_TO_TRAIN_ONLY_TESTS="online_ST_files/translated_tests_train_only.json"

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--langs",
        type=str,
        nargs="+",
        help="List of langs",
        default=["python", "java"],
    )

    parser.add_argument(
        "--obfuscate",
        type=bool_flag,
        default=False,
        help="obfuscate identifiers in source, target, and in validation / test sets",
    )

    parser.add_argument(
        "--add_spans",
        type=bool_flag,
        default=True,
        help="add spans for tree-structure in source, target, and in validation / test sets",
    )

    parser.add_argument(
        "--input_df_path",
        type=str,
        required=True,
        help="path to translation and test results df",
    )

    parser.add_argument(
        "--project_root",
        type=str,
        required=True,
        help="path_to_project_root",
    )

    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="output directory",
    )

    parser.add_argument(
        "--process_evaluation_set",
        type=bool_flag,
        default=False,
        help="process evaluation set",
    )

    parser.add_argument(
        "--process_df",
        type=bool_flag,
        required=True,
        help="whether or not to re-process the dataframe",
    )

    parser.add_argument(
        "--evaluation_set_directory",
        type=str,
        required=True,
        help="evaluation set directory",
    )

    parser.add_argument(
        "--make_offline_training_dataset",
        type=bool_flag,
        default=False,
        help="make offline training dataset",
    )

    parser.add_argument(
        "--make_online_training_dataset",
        type=bool_flag,
        default=False,
        help="make online training dataset",
    )

    parser.add_argument(
        "--bpe_codes_path",
        type=str,
        required=False,
        help="bpe codes path",
    )

    parser.add_argument(
        "--bpe_vocab_path",
        type=str,
        required=False,
        help="bpe vocab path",
    )

    return parser


def process_evaluation_set(eval_set_dir: Path, output_dir: Path, langs: List[str], obfuscate: bool, add_spans: bool,
                           bpe_model: BPEMode, vocab_path: Path, to_bin: bool = True):
    splits = ("valid", "test")
    langs.append("java")
    set_lang_data = {
        split: {
            lang: eval_set_dir.joinpath("transcoder_{}.{}.tok".format(split, lang)).open().readlines() for lang in langs
        } for split in splits
    }
    #pdb.set_trace()
    for split in splits:
        for lang1, lang2 in itertools.product(langs, langs):
            print(f"lang 1 {lang1}, lang2 {lang2} and continue {not (lang1<lang2)}")
            if not (lang1 < lang2):
                continue
            print("processing {} {} {}".format(split, lang1, lang2))
            l1_pth = output_dir.joinpath("{}.{}_sa-{}_sa.{}_sa.bpe".format(split, lang1, lang2, lang1))
            l2_pth = output_dir.joinpath("{}.{}_sa-{}_sa.{}_sa.bpe".format(split, lang1, lang2, lang2))
            with l1_pth.open("w") as l1_fh, l2_pth.open("w") as l2_fh:
                pbar = tqdm(total = len(set_lang_data[split][lang1]), desc = f"{lang1} {lang2}")
                n_failures = 0
                for l1_line, l2_line in zip(set_lang_data[split][lang1], set_lang_data[split][lang2]):
                    try:
                        l1_line = reformat_line(l1_line, lang1, obfuscate, add_spans, bpe_model)
                        l2_line = reformat_line(l2_line, lang2, obfuscate, add_spans, bpe_model)
                        l1_fh.write(l1_line.strip() + "\n")
                        l2_fh.write(l2_line.strip() + "\n")
                    except (ValueError, RuntimeError, KeyError, SyntaxError, LexerError, AssertionError, JavaSyntaxError) as e: 
                        print(e, flush=True)
                        traceback.print_exc()
                        print(f"line 1 was {l1_line}\n\nline 2 was {l2_line}", flush=True)
                        n_failures+=1
                        pbar.set_description(f"{lang1} {lang2} failures {n_failures}")
                    pbar.update(1)
            if to_bin:
                for pth in (l1_pth, l2_pth):
                    XLM_preprocess(str(vocab_path.absolute()),
                            str(pth),
                            str(pth).replace(".bpe", ".pth"))
    # make symlinks for validation and test sets to offline dataset path
    # subprocess.run(["ln", "-s", str(output_dir.joinpath("*.pth")), str(output_dir.joinpath("offline_dataset"))],
    #                check=True, shell=True)
    return


def get_annotator(lang):
    if lang == "python":
        return PythonSpanAnnotator
    elif lang == "java":
        return JavaSpanAnnotator
    else:
        raise NotImplementedError("lang {} not yet supported".format(lang))


def reformat_line(line, lang, obfuscate, add_spans, bpe_model):
    try: 
        _id, prog_str = line.split(" | ", 1) 
    except ValueError as e: 
        print(e)
        traceback.print_exc()
        print(f"failed on line {line}")
        raise SyntaxError
    # apply bpe to the id
    _id = bpe_model.apply_bpe(_id)
    # annotate the program (obfuscate and or spans)
    annotator = get_annotator(lang)
    if add_spans:
        reformatted = (annotator(prog_str, obfuscate, bpe_model)
                       .get_annotated_prog_str(tokenized_style=True, apply_bpe=True))
    elif obfuscate:
        reformatted = annotator(prog_str, obfuscate, bpe_model).get_prog(tokenized_style=True, apply_bpe=True)
    else:
        reformatted = prog_str
    return _id.strip() + " | " + reformatted.strip()

def dedup_df(input_df_path, output_df_path):
    df = pd.read_csv(input_df_path)
    df = df[~df.duplicated(subset=["TARGET_CLASS"])]
    df.to_csv(output_df_path, index=False)

def main(args):
    parser = get_parser()
    args = parser.parse_args()

    if args.make_online_training_dataset:
        assert args.make_offline_training_dataset, "make_offline_training_dataset not true for online"

    ## get from args
    output_directory_str = args.output_directory
    if not os.path.exists(output_directory_str): 
        os.makedirs(output_directory_str)
    output_dir = Path(output_directory_str)
    obfuscate = args.obfuscate
    add_spans = args.add_spans
    langs = args.langs

    project_root = args.project_root
    input_df_path = args.input_df_path
    bpe_codes_path = args.bpe_codes_path
    bpe_vocab_path = args.bpe_vocab_path
    #pdb.set_trace()
    if args.process_evaluation_set:
        eval_set_dir = Path(args.evaluation_set_directory)
        assert eval_set_dir and eval_set_dir.exists()
        assert args.bpe_codes_path and args.bpe_vocab_path
        assert Path(args.bpe_codes_path).exists() and Path(args.bpe_vocab_path).exists()
        bpe_model = FastBPEMode(
            codes=args.bpe_codes_path, vocab_path=args.bpe_vocab_path
        )
        print("process evaluation set")
        process_evaluation_set(eval_set_dir, output_dir, langs, obfuscate, add_spans, bpe_model, Path(bpe_vocab_path))

    for lang in langs:
            if lang == "java":
                continue
            output_df_path = os.path.join(output_directory_str, f"test_results_{lang}_df.csv")

            if (obfuscate | add_spans) and args.process_df:
                subprocess.run(["python",
                                    os.path.join(project_root, "codegen_sources/test_generation/reformat_df_translations.py"),
                                    "--input_df", input_df_path,
                                    "--output_df", output_df_path,
                                    "--obfuscate", str(obfuscate),
                                    "--spans", str(add_spans)], check=True)
                print("saved df to: ", output_df_path)
            else:
                print("copying df to: ", output_df_path)
                dedup_df(input_df_path, output_df_path)

            if args.make_offline_training_dataset:
                # if they exist they ideally should be the "test set" that was substututed from another script
                offline_dataset_path = os.path.join(output_directory_str, "offline_dataset/")
                try: 
                    subprocess.run(["python",
                                os.path.join(project_root,
                                             'codegen_sources/test_generation/select_successful_tests.py'),
                                "--input_df", os.path.dirname(output_df_path),
                                "--output_folder", offline_dataset_path,
                                "--bpe_path", bpe_codes_path, "--langs", "python", 
                                "--bpe_vocab", bpe_vocab_path], check=True)
                except subprocess.CalledProcessError as e: 
                    output = e.output
                    print("process failed with output: ", output)
                    exit(0)
                if args.make_online_training_dataset:
                    print("make online training dataset")
                    online_dataset_path = os.path.join(output_directory_str, "online_ST_files/")
                    move_files = False
                    if os.path.exists(os.path.join(online_dataset_path, ROOT_DIR_TO_TESTS)):
                        shutil.move(os.path.join(online_dataset_path, ROOT_DIR_TO_TESTS),
                                    os.path.join(online_dataset_path, ROOT_DIR_TO_TESTS_TEMP))
                        move_files = True
                    subprocess.run(["python",
                                os.path.join(args.project_root,
                                             'codegen_sources/test_generation/create_data_for_online_st.py'),
                                "--dataset_path", offline_dataset_path,
                                "--input_dfs_path", os.path.dirname(output_df_path),
                                "--output_path", online_dataset_path,
                                "--vocab_path", bpe_vocab_path], check=True)
                    if move_files:
                        # the subprocess will put the training set now into the ROOt_DIR_TO_TESTS, we want to move that to TRAIN_ONLY
                        # and then put the tmp back into the path that will be called during model training
                        shutil.move(os.path.join(online_dataset_path, ROOT_DIR_TO_TESTS), os.path.join(online_dataset_path, ROOT_DIR_TO_TRAIN_ONLY_TESTS))
                        shutil.move(os.path.join(online_dataset_path, ROOT_DIR_TO_TESTS_TEMP), os.path.join(online_dataset_path, ROOT_DIR_TO_TESTS))

    return 0

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

