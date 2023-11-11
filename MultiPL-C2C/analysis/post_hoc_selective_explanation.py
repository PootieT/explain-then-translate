import itertools
import os.path
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd

from src.single_experiment_pass_k import for_file


def get_dump_success(dump_dir):
    return np.array([for_file(p) for p in itertools.chain(Path(dump_dir).glob("*.results.json"))])[:, 0]


def selective_aggregate(baseline_dump: str, exp_dump: str, success_threshold: float=0.9):
    out_dump = exp_dump.replace("-completion", f"-post_hoc_selective_{success_threshold:.1f}-completion")
    # if not os.path.exists(out_dump):
    os.makedirs(out_dump, exist_ok=True)
    success = get_dump_success(baseline_dump)
    names = [json.load(open(p))["name"] for p in itertools.chain(Path(baseline_dump).glob("*.results.json"))]
    exp_cnt = 0
    for name, suc in zip(names, success):
        tgt_dir = f"{out_dump}/{name}.results.json"
        if suc >= success_threshold:
            # if problem is easy enough, don't explain, translate directly
            src_dir = f"{baseline_dump}/{name}.results.json"
        else:
            exp_cnt += 1
            src_dir = f"{exp_dump}/{name}.results.json"
        shutil.copy(src_dir, tgt_dir)
    exp_success = get_dump_success(exp_dump)
    result_success = get_dump_success(out_dump)
    direction = "-".join(baseline_dump.split("/")[-1].split("-")[1:3])
    print(f"{direction}, exp_cnt={exp_cnt}/{len(names)}, "
          f"direct={success.mean():.3f}, exp={exp_success.mean():.3f}, select={result_success.mean():.3f}")
    return {
        "direction": direction, "exp_cnt": exp_cnt, "total": len(names), "threshold": success_threshold,
        "direct": success.mean(), "exp": exp_success.mean(), "select": result_success.mean()
    }

def loop_over_exp(dump_root: str, threshold: float=0.9):
    root = Path(__file__).parents[1].joinpath(dump_root)
    rows = []
    for direction in os.listdir(root):
        if "-" not in direction:
            continue
        baseline_dump = str(root.joinpath(f"{direction}/humaneval-{direction}-PTremove-completion"))
        exp_dump = str(root.joinpath(f"{direction}/humaneval-{direction}-PTremove-MTexplain-completion"))
        row = selective_aggregate(baseline_dump, exp_dump, threshold)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(root.parent.joinpath(f"translation_results/post_hot_selection_{threshold}.csv"))


if __name__=="__main__":
    loop_over_exp("dump_chatgpt", threshold=0.95)