import pdb
import os
import numpy as np
from pathlib import Path
import json
import itertools
import argparse
import gzip

def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def open_json(fpath: Path, mode: str):
    return gzip.open(fpath, mode + "t") if fpath.suffix == ".gz" else open(fpath, mode)

def for_file(path):
    with open_json(path, "r") as f:
        try:
            data = json.load(f)
        except:
            raise Exception(f"Error loading json: {path}")
    n = len(data["results"])
    try:
        c = len([True for r in data["results"] if r["status"] == "OK" and r["exit_code"] == 0])
    except:
        print(f"corrupt result file: {path}, deleting")
        os.remove(path)
        exit()
    return np.array([estimator(n, c, 1), estimator(n, c, 5), estimator(n, c, 10)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, help="Temperature completions were made at. \
                        If 0.2 runs pass@1 rather than pass@10 and pass@100", required=True)
    parser.add_argument("dirs", type=str,  help="Directories with results. ", nargs="+")
    args = parser.parse_args()
    k = 1 if args.temperature == 0.2 else 10
    print("Dataset,Pass@k,Estimate")
    for d in args.dirs:
        result_array = np.array([ for_file(p) for p in itertools.chain(Path(d).glob("*.results.json"), Path(d).glob("*.results.json.gz")) ])
        if len(result_array) < k:
            continue
        result = result_array.mean(axis=0)
        name = d.split("/")[-1] if d.split("/")[-1] != "" else d.split("/")[-2]
        if args.temperature == 0.2:
            print(f"{name},1,{result[0]:.3f}")
        # else:
        print(f"{name},5,{result[1]:.3f}")
        print(f"{name},10,{result[2]:.3f}")


if __name__ == "__main__":
    main()
