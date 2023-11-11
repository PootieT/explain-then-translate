import argparse
import shutil
from codegen_sources.preprocessing.utils import bool_flag
import os
import json
import random
import pandas as pd
import pdb

ID_JOIN_FIELD="TARGET_CLASS"

ROOT_DIR_TO_TESTS="online_ST_files/translated_tests.json"
ROOT_DIR_TO_OLD_TESTS="online_ST_files/translated_tests_train_and_test.json"

ROOT_DIR_TO_DF="test_results_python_df.csv"
ROOT_DIR_TO_OLD_DF="test_results_python_df_train_and_test.csv"


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--root_dir_in", help="path to the input dataframe",
    )

    parser.add_argument(
        "--root_dir_out", help="path to the input dataframe",
    )

    parser.add_argument(
        "--overwrite",
        type=bool_flag,
        default=True,
        help="overwrite out dir if exists",
    )

    parser.add_argument(
        "--testcase_prop",
        type=float,
        default=0.05,
        help="proportion to sample for test set",
    )
    return parser

def read_json(fh): 
    res = []
    for line in fh: 
        res.append(json.loads(line))
    return res

def write_json(list_of_dicts, fh): 
    for d in list_of_dicts: 
        fh.write(json.dumps(d) + "\n")
    return 0


def subset_and_write_testcases(root_dir, testcase_prop):
    pdb.set_trace()
    path_to_testcases = os.path.join(root_dir, ROOT_DIR_TO_TESTS)
    path_to_testcases_old = os.path.join(root_dir, ROOT_DIR_TO_OLD_TESTS)
    #shutil.copy2(path_to_testcases, path_to_testcases_old)
    testcase_json_list = read_json(open(path_to_testcases))
    testcase_json_list = [testcase for testcase in testcase_json_list if random.random() < testcase_prop]
    #with open(path_to_testcases, "w") as f:
    #    write_json(testcase_json_list, f)

    test_ids = set([testcase[ID_JOIN_FIELD] for testcase in testcase_json_list])

    return test_ids


def subset_and_write_df(root_dir, test_ids):
    path_to_df = os.path.join(root_dir, ROOT_DIR_TO_DF)
    path_to_df_old = os.path.join(root_dir, ROOT_DIR_TO_OLD_DF)
    shutil.copy2(path_to_df, path_to_df_old)
    df = pd.read_csv(path_to_df)
    df = df[~df[ID_JOIN_FIELD].isin(test_ids)]
    df = df[~df.duplicated(subset=ID_JOIN_FIELD)] # remove duplicates
    df.to_csv(path_to_df, index=False)
    return None


def copy_files(root_dir_in, root_dir_out, overwrite):
    if os.path.exists(root_dir_out) and not overwrite:
        raise Exception("out dir exists and overwrite flag is set to false")
    if os.path.exists(root_dir_out) and overwrite: 
        shutil.rmtree(root_dir_out)
    shutil.copytree(root_dir_in, root_dir_out)
    shutil.rmtree(os.path.join(root_dir_out, "offline_dataset"))
    return 0


def main():
    parser = parse_arguments()
    args = parser.parse_args()
    copy_files(args.root_dir_in, args.root_dir_out, args.overwrite)
    test_ids = subset_and_write_testcases(args.root_dir_out, args.testcase_prop)
    return subset_and_write_df(args.root_dir_out, test_ids)


if __name__ == "__main__":
    main()
