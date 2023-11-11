import argparse
import os.path

import pandas as pd
import json
from subset_testcases_and_df import read_json, write_json
import shutil
from tqdm import tqdm

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="augment_translated_tests")

    # main parameters
    parser.add_argument(
        "--path_to_translated_tests_json", type=str, help="path to translated tests json"
    )

    # main parameters
    parser.add_argument(
        "--path_to_translated_tests_df", type=str, help="path to translated tests csv"
    )

    parser.add_argument(
        "--output_path", type=str, help="path for results json"
    )
    return parser

def main():
    """
    Main function.
    """
    # get parameters
    args = get_parser().parse_args()
    print(args)

    # load data
    df = pd.read_csv(args.path_to_translated_tests_df)
    df = df.set_index("TARGET_CLASS")
    translated_tests_dicts = read_json(open(args.path_to_translated_tests_json))
    for d in tqdm(translated_tests_dicts, desc="augmeting all dict elements"):
        row = df.loc[d["TARGET_CLASS"]]
        d["tests_strings"] = row["tests_strings"]
        d["java_function"] = row["java_function"]
    if args.output_path == args.path_to_translated_tests_json:
        backup_path = os.path.splitext(args.path_to_translated_tests_json)[0] + "_backup.json"
        shutil.move(args.path_to_translated_tests_json, backup_path)
    write_json(translated_tests_dicts, open(args.output_path))
    print("Done. Results written to {}".format(args.output_path))

if __name__ == "__main__":
    main()


