import argparse
import os
import json
from typing import *
import itertools
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd

from analysis.plot_translation_results import get_bucket_col_names
from src.single_experiment_error_types import understand_errors, get_all_data, get_result_breakdown_by_len, ERROR_TYPES
from src.single_experiment_pass_k import for_file


def get_exp_params(path: str) -> Dict[str, Any]:
    attr = {}
    file_name = Path(path).name
    attr["shots"] = 4 if "-4shot" in file_name else 0
    if "explain" not in file_name:
        attr["exp"] = "baseline"
    elif "explain-lbl-simp" in file_name:
        attr["exp"] = "exp-lbl-d"
    elif "explain-lbl" in file_name:
        attr["exp"] = "exp-lbl"
    else:
        attr["exp"] = "exp"
    return attr


def collect(dump_dir:str,detail=False):
    result_path = Path(__file__).resolve().absolute().parents[1].joinpath(
        f"translation_results/{Path(dump_dir).name}.csv")
    rows = []
    for lang_pair in tqdm(os.listdir(dump_dir)):
        if "-" not in lang_pair:  # garbage files/directories
            continue
        for exp in os.listdir(f"{dump_dir}/{lang_pair}"):
            exp_path = f"{dump_dir}/{lang_pair}/{exp}"
            results = get_exp_params(exp_path)

            # sanity check: go around and delete any .results.json if the eval results are incomplete
            any_corrupt = 0
            for p in Path(exp_path).glob("*.results.json"):
                with open(p) as f:
                    data = json.load(f)
                if any([len(res)==0 for res in data["results"]]):
                    os.remove(p)
                    any_corrupt+=1
            if any_corrupt:
                print(f"found {any_corrupt} corrupt results file in {exp_path}. existing now.")
                exit()

            # pass@k score
            result_array = np.array([for_file(p) for p in Path(exp_path).glob("*.results.json")])
            if len(result_array) < 1:
                continue
            scores = result_array.mean(axis=0)
            results.update({
                "src_lang": lang_pair.split("-")[0],
                "tgt_lang": lang_pair.split("-")[1],
                "num_probs": len(result_array),
                "Pass@1(n=20)": round(scores[0], 3),
                "Pass@5(n=20)": round(scores[1], 3),
                "Pass@10(n=20)": round(scores[2], 3),
            })

            # error breakdown by type and by length
            try:
                result_array = np.array([understand_errors(p) for p in get_all_data(exp_path)]).squeeze()
                errors = result_array.mean(axis=0)
                for err_type, err in zip(ERROR_TYPES, errors):
                    results[err_type] = err
            except Exception as e:
                print(f"exception getting error breakdown: {e}")

            try:
                result_by_len = get_result_breakdown_by_len(result_array, get_all_data(exp_path))
                len_bins = get_bucket_col_names(4)
                for bin_name, err in zip(len_bins, result_by_len):
                    results[bin_name] = err
            except Exception as e:
                print(f"exception getting error by length breakdown: {e}")

            rows.append(results)

    df = pd.DataFrame(rows)
    df.to_csv(result_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str,  help="Directory with results. ")
    args = parser.parse_args()
    collect(args.dir)


if __name__ == "__main__":
    # collect("../dump_codegen216b")
    main()