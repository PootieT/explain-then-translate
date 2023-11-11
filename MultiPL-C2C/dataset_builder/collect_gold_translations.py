import collections
import os
import json
from typing import *

import numpy as np

from dataset_builder.generic_translator import extract_python_source_code
from dataset_builder.utils import FEW_SHOT_EXAMPLES


def extract_python_program(python_path: str):
    reading_prompt = True
    reading_tests = False
    reading_source_program = False
    prompt_buffer = []
    tests_buffer = []
    source_program_buffer = []
    with open(python_path) as f:
        for line in f:
            if "### Canonical solution below ###" in line:
                reading_prompt = False
                reading_source_program = True
            if "### Unit tests below ###" in line:
                reading_tests = True
                reading_source_program = False
                continue
            if "def test_check():" in line:
                break

            if reading_prompt:
                prompt_buffer.append(line)
            if reading_tests:
                tests_buffer.append(line)
            if reading_source_program:
                source_program_buffer.append(line)
    code = extract_python_source_code("remove", prompt_buffer, "remove", source_program_buffer, "keep")
    return code


def collect_python_programs(data_dir: str, out_dir: str):
    data = {}
    for f in os.listdir(data_dir):
        name = f.replace(".py", "")
        code = extract_python_program(f"{data_dir}/{f}")
        data[name] = code

    print(f"tgt lang = py, update_count = 0, total # solutions = {len(data)}")
    with open(f"{out_dir}/py.json", "w") as f:
        json.dump(data, f)


def get_shortest_pass_program(results: List[Dict[str, str]]) -> str:
    best_program = None
    for result in results:
        if result["status"] == "OK":
            if best_program is None:
                best_program = result["program"]
            elif len(result["program"]) < len(best_program):
                best_program = result["program"]
    return best_program


def get_wrong_programs_counter(results: List[Dict[str, str]], gold_program: Optional[str]=None, all_programs=False) -> collections.Counter:
    if all_programs:
        program_counter = Counter([res["program"] for res in results])
    else:
        program_counter = Counter([res["program"] for res in results if res["status"]!= "OK"])
    if (not all_programs) and (gold_program is not None) and (gold_program in program_counter):
        print(f"weird, gold program included in worng program dict")
        del program_counter[gold_program]
    return program_counter


def collect_over_lang_dir(data_dir: str, out_dir: str, dataset_name: str="humaneval"):
    src_lang, tgt_lang = data_dir.split("-")
    src_lang = "py" if src_lang == "python" else src_lang
    out_path = f"{out_dir}/{tgt_lang}.json"
    if os.path.isfile(out_path):
        solutions = json.load(open(out_path))
    else:
        os.makedirs(out_dir, exist_ok=True)
        solutions = {}

    update_count = 0
    for exp in os.listdir(data_dir):
        exp_dir = f"{data_dir}/{exp}"
        if "completion" not in exp_dir:  # chatGPT outputs are messy, not good for prompting
            continue
        for f in os.listdir(exp_dir):
            if not f.endswith("results.json") and dataset_name not in f:
                continue
            data_path = f"{exp_dir}/{f}"
            data = json.load(open(data_path))
            best_passed_program = get_shortest_pass_program(data["results"])
            if best_passed_program is None:
                continue
            if data["name"] not in solutions:
                solutions[data["name"]] = best_passed_program
                update_count += 1
            elif len(solutions[data["name"]]) > len(best_passed_program):
                solutions[data["name"]] = best_passed_program
                update_count += 1
    print(f"tgt lang = {tgt_lang}, update_count = {update_count}, total # solutions = {len(solutions)}")
    json.dump(solutions, open(out_path, "w"), indent=2)


def collect_wrong_programs_over_lang_dir(data_dir: str, out_dir: str, dataset_name: str="humaneval", all_programs=False):
    src_lang, tgt_lang = data_dir.split("-")
    src_lang = "py" if src_lang == "python" else src_lang
    out_path = f"{out_dir}/{tgt_lang}.json"
    # TODO we can't reload because counters would just keep increasing
    # if os.path.isfile(out_path):
    #     solutions = json.load(open(out_path))
    # else:
    if not all_programs:
        assert os.path.isfile(out_path.replace("_wrong", "")), "need to load gold solution just to make sure no " \
                                                               "programs there is included in this dataset"
        gold_solutions = json.load(open(out_path.replace("_wrong", "")))
    else:
        gold_solutions = {}

    os.makedirs(out_dir, exist_ok=True)
    solutions = {}

    for exp in os.listdir(data_dir):
        exp_dir = f"{data_dir}/{exp}"
        if ("completion" not in exp_dir) or \
            any([w in exp_dir for w in ["obf", "pivot", "ALT", "retrieval"]]):   #  # chatGPT outputs are messy, and obf programs are not the same, not good for prompting
            continue
        for f in os.listdir(exp_dir):
            if not f.endswith("results.json") and dataset_name not in f:
                continue
            data_path = f"{exp_dir}/{f}"
            data = json.load(open(data_path))
            wrong_programs_counter = get_wrong_programs_counter(data["results"], gold_solutions.get(data["name"]), all_programs)
            if len(wrong_programs_counter) == 0:
                continue
            if data["name"] not in solutions:
                solutions[data["name"]] = wrong_programs_counter
            else:
                solutions[data["name"]].update(wrong_programs_counter)
    cnts = np.array([len(v) for v in solutions.values()])
    print(f"tgt lang={tgt_lang}, average solns={cnts.mean():.2f}, max solns={max(cnts)}, min solns={min(cnts)}")
    json.dump(solutions, open(out_path, "w"), indent=2)


def collect(dump_dir: str, out_dir: str, dataset_name: str, program_subset: str="gold"):
    for lang_dir in os.listdir(dump_dir):
        if not lang_dir.startswith("py"):
            continue
        print(f"collecting {lang_dir} ....")
        if program_subset == "gold":
            collect_over_lang_dir(f"{dump_dir}/{lang_dir}", out_dir, dataset_name)
        elif program_subset == "wrong":
            out_dir += "_wrong" if not out_dir.endswith("_wrong") else ""
            collect_wrong_programs_over_lang_dir(f"{dump_dir}/{lang_dir}", out_dir, dataset_name)
        elif program_subset == "all":
            out_dir += "_all" if not out_dir.endswith("_all") else ""
            collect_wrong_programs_over_lang_dir(f"{dump_dir}/{lang_dir}", out_dir, dataset_name, all_programs=True)
        else:
            raise NotImplementedError("choose one of following program subset: {gold, wrong, all}")

    if not os.path.isfile(f"{out_dir}/py.json") and program_subset == "gold":
        py_dir = "../datasets/originals" if dataset_name == "humaneval" else f"../datasets/{dataset_name}"
        collect_python_programs(py_dir, out_dir)


if __name__ == "__main__":
    collect(
        dump_dir="../dump",
        out_dir="../datasets/humaneval_multi",
        dataset_name="humaneval",
        program_subset="wrong"
    )

