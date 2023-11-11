import collections
import json
import itertools
import os.path
from collections import Counter
from pprint import pprint

from tqdm import tqdm
import regex as re
from pathlib import Path
import gzip
from typing import List, Optional, Union

import pandas as pd
import numpy as np

from dataset_builder.utils import find_all, get_method_name, get_source_code_from_prompt, SHORT2CANONICAL
from src.single_experiment_error_types import *


def open_json(fpath: Path, mode: str):
    return gzip.open(fpath, mode + "t") if fpath.suffix == ".gz" else open(fpath, mode)


def extract_src_code(translation_prompt):
    if isinstance(translation_prompt, list):
        translation_prompt = translation_prompt[-1]["content"]
    src_lang = translation_prompt[translation_prompt.find("### ")+4:].split("\n")[0].lower()
    start = list(find_all(translation_prompt.lower(), f"### {src_lang}"))[-1]
    end = translation_prompt.find("### ", start + 1)
    return translation_prompt[start: end]


def read_folder(data_dir):
    df = []
    # transcoder evaluation
    if os.path.exists(data_dir+"_aggregated_output.json"):
        eval_data = json.load(open(data_dir+"_aggregated_output.json"))
        eval_data = [{"name": k, "results": v} for k,v in eval_data.items()]
        for row in eval_data:
            row["results"] = [{
                "status": "OK" if result[0] == "success" else result[0],
                "stderr": result[1] if result[0] != "success" else "",
                "stdout": result[1] if result[0] == "success" else "",
            } for result in row["results"]]
            beam1_features = {f"beam1_{k}": v for k, v in row["results"][0].items()}
            data = json.load(open(f"{data_dir}/{row['name']}.json"))
            row.update(beam1_features)
            row.update(data)
            row["beam1_program"] = data["prompt"] + data["completions"][0]

        df = pd.DataFrame(eval_data)
    else:  # MultiPL-E eval
        for path in itertools.chain(Path(data_dir).glob("*.results.json"), Path(data_dir).glob("*.results.json.gz")):
            with open_json(path, "r") as f:
                data = json.load(f)
                beam1_features = {f"beam1_{k}": v for k, v in data["results"][0].items()}
                data.update(beam1_features)
                # del data["results"]
            df.append(data)
        df = pd.DataFrame(df)

    df = df.sort_values("name")
    df = df.set_index("name")
    return df


def print_non_assertion_errors(df, show_src=True, error_type=2, show_intermediate=False):
    cnt = 0
    lang = df["language"][0]
    get_error_func = globals().get(f"get_{lang}_error_types", globals().get(f"get_{SHORT2CANONICAL[lang]}_error_types"))
    for i, row in df.iterrows():
        # if "Optional.empty()" in row["beam1_program"]:
        if get_error_func(row["results"])[0] == error_type:
        # if row["beam1_status"] != "OK": # and len(row["beam1_stderr"]) == 0:
        # if len(row["beam1_stderr"]) > 0 and "AssertionError" not in row["beam1_stderr"]:
            # if "Pair" in row["prompt"]:# and "org/javatuples/Pair" not in row["beam1_stderr"]:
            # if "cannot find symbol" in row["beam1_stderr"]:
            print(f"======== {i} ========")
            if show_src:
                print(extract_src_code(row["translation_prompt"]))
            if show_intermediate:
                print(get_source_code_from_prompt(row["translation_prompt"]))
            print(row["beam1_program"])
            print("STDERR: ", row["beam1_stderr"])
            print("STDOUT: ", row["beam1_stdout"])
            cnt += 1
    print(f" ***** total of {cnt} programs *****  ")


def get_prompt_str(translation_prompt, section=1):
    if isinstance(translation_prompt, str):
        indices = list(find_all(translation_prompt.lower(), "### python"))
        start = indices[-1]
        end = translation_prompt.find("###", start + 1)
        while section > 0:
            start = end
            end = translation_prompt.find("###", start + 1)
            section -= 1
        return translation_prompt[start: end]
    else:
        return translation_prompt[-2]["content"]


def print_error_first_then_success(df1, df2, show_steps=False):
    cnt = 0
    for i, row in df1.iterrows():
        if i not in df2.index:  # different setting maybe testing different subsets
            continue
        row2 = df2.loc[i]
        if row["beam1_status"] != "OK" and row2["beam1_status"]=="OK":
            # if "Pair" in row["prompt"]:# and "org/javatuples/Pair" not in row["beam1_stderr"]:
            # if "cannot find symbol" in row["beam1_stderr"]:
            print(f"======== {i} ========")
            if show_steps:
                print(" ----> bad steps")
                print(get_prompt_str(row["translation_prompt"]))
                print(" ----> good steps")
                print(get_prompt_str(row2["translation_prompt"]))

            # print(extract_src_code(row["translation_prompt"]))
            print(row["beam1_program"])
            print("STDERR: ", row["beam1_stderr"])
            print("STDOUT: ", row["beam1_stdout"])
            print(row2["beam1_program"])
            cnt += 1
    print(f" ***** total of {cnt} programs *****  ")


def print_no_success(df, show_src=True):
    cnt = 0
    for i, row in df.iterrows():
        if -1 not in get_java_error_types(row["results"]):
            print(f"======== {i} ========")
            if show_src:
                print(extract_src_code(row["translation_prompt"]))
            print(row["beam1_program"])
            print(row["beam1_stderr"])
            cnt += 1
    print(f" ***** total of {cnt} programs *****  ")


def print_union_success_vs_individual_succes(dirs: List[str], all_beams=False):
    union_success = set([])
    for d in dirs:
        df = read_folder(d)
        if all_beams:
            success_df = df[df.results.apply(lambda res: any([r["status"]=="OK" for r in res]))]
        else:
            success_df = df[df["beam1_status"] == "OK"]
        print(f"dir={d}, success={len(success_df)/len(df):.3f}")
        union_success = union_success.union(success_df.index)
    print("="* 20)
    print(f"union, success={len(union_success)/len(df):.3f}")


def print_intermediate_step_influence(basedir: str, interdir: str, combine_dir: str):
    """
    define base_success
    calculate:
        - comb success rate if intermediate step is correct / incorrect
        -
    """
    base_df, inter_df, comb_df = read_folder(basedir), read_folder(interdir), read_folder(combine_dir)
    all_ids = set(base_df.index)
    base_success = set(base_df[base_df["beam1_status"] == "OK"].index)
    inter_success = set(inter_df[inter_df["beam1_status"] == "OK"].index)
    inter_fail = all_ids.difference(inter_success)
    comb_success = set(comb_df[comb_df["beam1_status"] == "OK"].index)
    print("success rate with correct intermediate steps (after combining)")
    print(len(comb_success.intersection(inter_success))/len(all_ids.intersection(inter_success)))
    print("success rate with incorrect intermediate steps (after combining)")
    print(len(comb_success.intersection(inter_fail)) /len(all_ids.intersection(inter_fail)))
    print("success rate with correct intermediate steps (baseline)")
    print(len(base_success.intersection(inter_success)) / len(all_ids.intersection(inter_success)))
    print("success rate with incorrect intermediate steps (baseline)")
    print(len(base_success.intersection(inter_fail)) / len(all_ids.intersection(inter_fail)))


def print_debug_influence(base_dir: str, debug_dir: str, ):
    """
    define base_success
    calculate:
        - comb success rate if intermediate step is correct / incorrect
        -
    """
    base_df, debug_df = read_folder(base_dir), read_folder(debug_dir)
    all_ids = set(base_df.index)
    base_success = set(base_df[base_df["beam1_status"] == "OK"].index)
    base_fail = all_ids.difference(base_success)
    debug_success = set(debug_df[debug_df["beam1_status"] == "OK"].index)
    print(f"base success rate: {len(base_success)/len(all_ids)}")
    print("success rate with correct initial translation")
    print(len(debug_success.intersection(base_success))/len(base_success))
    print("success rate with incorrect initial translation")
    print(len(debug_success.intersection(base_fail)) /len(base_fail))
    print("Total success rate if only debug on failed example")
    print(len(debug_success.union(base_success)) / len(all_ids))
    lines_changed = pd.Series([get_percent_lines_changed(
        base_df.loc[n]["beam1_program"], debug_df.loc[n]["beam1_program"]) for n in all_ids])
    print("similarity statistics")
    print(lines_changed.describe())
    print(f"no change percent: {(lines_changed == 0).sum()/len(all_ids)}")


def get_percent_lines_changed(program, new_program):
    program_lines = set([l.strip() for l in program.split("\n")])
    new_program_lines = set([l.strip() for l in new_program.split("\n")])
    return len(new_program_lines.difference(program_lines))


def count_func_calls(prompt_path: str):
    data = json.load(open(prompt_path))
    intermediate_count = 0
    num_calls = []
    counter = Counter()
    for problem in data:
        if "### Method" not in problem["translation_prompt"]:

            continue
        intermediate_count += 1
        first_sent = problem["translation_prompt"][problem["translation_prompt"].find("### Method call "):
                                                   problem["translation_prompt"].find("Let's translate them")]
        func_calls = re.findall(r"\`[^`]*\`", first_sent)
        methods = [get_method_name(c.replace("`","")) for c in func_calls]
        num_calls.append(len(methods))
        counter.update(methods)
    print(f"Number of program with rare func_calls: ({intermediate_count}/{len(data)})({intermediate_count/len(data):.3f})")
    print(f"Number of function call stats: {pd.Series(num_calls).describe()}")
    print(f"Function Call counters:")
    pprint(counter)


def print_multi_beam_as_debug_result(base_dir: str):
    base_df = read_folder(base_dir)
    num_generations = len(base_df.results[0])
    non_success_ids = set(base_df.index)
    for i in range(num_generations):
        total_df = base_df[base_df.index.isin(non_success_ids)]
        success_ids = total_df[total_df.results.apply(lambda x: x[i]["status"]) == "OK"].index
        non_success_ids = non_success_ids.difference(success_ids)
        total_success_fraction = (len(base_df)-len(non_success_ids)) / len(base_df)
        print(f"Up till beam {i+1}, pseudo-debug passes {total_success_fraction:.3f}")


def get_problem_success_rate(df, pid, subset: Union[int, List]):
    row = df.loc[pid]
    subset = subset if isinstance(subset, list) else list(range(subset))
    subset = [i for i in subset if i < len(row.results)]
    if len(subset) == 0:
        return -1
    success_count = sum([res["status"] == "OK" for res in np.array(row.results)[subset]])
    return success_count / len(subset)


def analyze_intermediate_goodness(data_dirs: List[str], base_dir: str, print_src: bool=True):
    # only look at problems that have low success rates
    base_df = read_folder(base_dir)
    dfs = [read_folder(d) for d in data_dirs]
    any_fails = base_df.results.apply(lambda x: any([beam_res["status"] != "OK" for beam_res in x]))
    hard_prob_ids = base_df[any_fails].index
    hard_prob_ids = [i for i in hard_prob_ids if i in dfs[0].index]  # only consider those included in experiment df trials
    max_beam = min([len(df.results[0]) for df in dfs])

    # for each problem, rank the intermediate generation based on how many fractions of completion passes
    ranked_intermediate_steps = {}
    for pid in hard_prob_ids:
        success_rates = [get_problem_success_rate(df, pid, max_beam) for df in dfs]
        rank_idx = np.argsort(success_rates)[::-1]
        ranked_intermediate_steps[pid] = {
            "success_rates": [success_rates[i] for i in rank_idx],
            "generations": [get_prompt_str(dfs[i].loc[pid].translation_prompt).strip() for i in rank_idx]
        }

    for name, ranked_steps in ranked_intermediate_steps.items():
        print(f"============= {name}: success_rates={ranked_steps['success_rates']} =============")
        if sum(ranked_steps['success_rates'])/len(ranked_steps['success_rates']) in [0.0, 1.0]:
            print("uninteresting example, skip")
            continue
        if print_src:
            print(get_prompt_str(base_df.loc[name].translation_prompt,0))
        for i, step in enumerate(ranked_steps["generations"]):
            print(f"---------- rank {i}: success rate={ranked_steps['success_rates'][i]} ----------")
            print(step)

    # if we were to assumed to have picked the best explanation, what's the best aggregated performance we can get
    def get_possible_success_rates(intermediate_rank: Optional[int]=None):
        success_cnt = len(base_df) - len(hard_prob_ids)
        for res in ranked_intermediate_steps.values():
            if intermediate_rank is None:
                thresh = res["success_rates"][np.random.randint(0,len(res["success_rates"]))]
            else:
                thresh = res["success_rates"][intermediate_rank]
            if np.random.rand() <= thresh:
                success_cnt += 1
        return success_cnt
    print("")
    print(f"Best possible success rate@1: {get_possible_success_rates(0)/len(base_df):.3f}")
    print(f"Random success rate@1: {get_possible_success_rates(None) / len(base_df):.3f}")
    print(f"Worst possible success rate@1: {get_possible_success_rates(-1)/len(base_df):.3f}")


def assert_no_tgt_lang_specific_explanation(dump_dir: str, tgt_lang: str = "Java"):
    df = read_folder(dump_dir)
    bad_cnt = 0
    for i, row in df.iterrows():
        if isinstance(row.translation_prompt, list):
            exps = [get_source_code_from_prompt(p) for p in row.translation_prompt]
        else:
            exps = [get_source_code_from_prompt(row.translation_prompt)]
        prog_bad_cnt = 0
        new_exp = []
        for exp in exps:
            if any([w.lower() in exp.lower() for w in [tgt_lang, "rewrite"]]):
                prog_bad_cnt += 1
        if prog_bad_cnt > 0:
            print(f"====== {i}, bad_count={prog_bad_cnt} ======")
        bad_cnt += prog_bad_cnt
    print(f"total of {bad_cnt}")


def collect_program_difficulty_level(lang: str, dump_dir: str):
    success_cnt = collections.defaultdict(int)
    total_cnt = collections.defaultdict(int)
    for exp_dir in os.listdir(f"{dump_dir}/py-{lang}"):
        for f in os.listdir(f"{dump_dir}/py-{lang}/{exp_dir}"):
            if not f.endswith(".results.json"):
                continue
            file_path = f"{dump_dir}/py-{lang}/{exp_dir}/{f}"
            data = json.load(open(file_path))
            success_cnt[data["name"]] += len([res for res in data["results"] if res["status"]=="OK"])
            total_cnt[data["name"]] += len(data["results"])
    rate_cnt = {k: v/total_cnt[k] for k, v in success_cnt.items()}
    counter = collections.Counter(rate_cnt)
    for i, (k, v) in enumerate(counter.most_common(n=len(counter))[::-1]):
        print(f"least common #{i}: {k}, success rate: {v:.3f}")


def print_non_trivial_problems(base_dir: str):
    base_df = read_folder(base_dir)
    for i, row in base_df.iterrows():
        if "success_rates" in row:
            success_cnt = sum(row.success_rates)
        else:
            success_cnt = len([1 for res in row.results if res["status"]=="OK"])
        if success_cnt not in [0, len(row.results)]:
             print(f"{row.name}, success_rate: {success_cnt/len(row.results):.1f}")


def print_intermediate_length(base_dir: str, print_res = True):
    enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
    df = read_folder(base_dir)
    for i, row in df.iterrows():
        src = extract_src_code(row.translation_prompt)
        src = src[src.find("\n\n")+2:]
        exp = get_source_code_from_prompt(row.translation_prompt)
        src_len = len(enc.encode(src))
        exp_len = len(enc.encode(exp))
        df.loc[i, "src_len"] = src_len
        df.loc[i, "exp_len"] = exp_len
        df.loc[i, "exp_src_ratio"] = exp_len/src_len
    if print_res:
        print("src_len:")
        print(df.src_len.describe())
        print("exp_len:")
        print(df.exp_len.describe())
        print("src_len:")
        print(df.exp_src_ratio.describe())
    return df


def print_all_tgt_specific_exp(shot=0):
    dfs = []
    for l in tqdm(SHORT2CANONICAL.keys()):
        if l in ["java","py","go_test.go"]:
            continue
        dump_path = f"../dump/py-{l}/humaneval-py-{l}-PTremove-MTexplain-completion-not-java"
        if shot == 4:
            dump_path = dump_path.replace("explain", "explain-4shot")
        dfs.append(print_intermediate_length(dump_path, print_res=False))

    df = pd.concat(dfs)
    print("src_len:")
    print(df.src_len.describe())
    print("exp_len:")
    print(df.exp_len.describe())
    print("src_len:")
    print(df.exp_src_ratio.describe())


if __name__ == "__main__":
    # df = read_folder("../datasets/originals-comments-inline-s4")
    # df = read_folder("../dump/py-java/humaneval-py-java-PTremove-MTexplain-lbl-4shot-completion")
    # df = read_folder("../dump_codegen216b/py-js/humaneval-py-js-PTremove-MTexplain-completion")
    # df = read_folder("../dump/cpp-py/transcoder_fixed_eval-cpp-py-TSkeep-MTexplain-completion")
    # print_non_assertion_errors(df, show_src=True, error_type=2, show_intermediate=True)
    # print_no_success(df, show_src=False)

    l = "r"
    df1 = read_folder(f"../dump_chatgpt/py-{l}/humaneval-py-{l}-PTremove-completion")
    df2 = read_folder(f"../dump_chatgpt/py-{l}/humaneval-py-{l}-PTremove-MTexplain-completion")
    print_error_first_then_success(df1, df2, show_steps=True)
    # print_union_success_vs_individual_succes([
    #     "../dump/py-java/humaneval-py-java-PTremove-MTexplain-lbl-simp-4shot-completion",
    #     "../dump/py-java/humaneval-py-java-PTremove-MTfunc2-completion",
    #     # "../dump/python-java/humaneval-python-java-PTremove-MTsteps-specific-completion",
    #     # "../dump/python-java/humaneval-python-java-PTremove-MTsteps-completion",
    #     # "../dump/python-java/humaneval-python-java-PTremove-MTrewrite-completion",
    # ], all_beams=True)
    # print_intermediate_step_influence(
    #     basedir="../dump/python-java/humaneval-python-java-PTremove-completion",
    #     interdir="../dump/python-sh/humaneval-python-sh-PTremove-completion",
    #     combine_dir="../dump/python-java/humaneval-python-java-PTremove-MTmulti-view_sh_latex-completion",
    # )

    # count_func_calls("../translation_prompts/python-d/humaneval-python-d-PTremove-MTfunc-completion.json")

    # print_debug_influence(
    #     base_dir="../dump/py-java/humaneval-py-java-PTremove-completion",
    #     debug_dir="../dump/py-java/humaneval-py-java-PTremove-MTdebug-completion"
    # )
    # print_multi_beam_as_debug_result(
    #     base_dir="../dump/py-java/humaneval-py-java-PTremove-completion",
    # )

    # analyze_intermediate_goodness(
    #     data_dirs=[
    #         "../dump/py-java/humaneval-py-java-PTremove-MTexplain20RR-completion",
    #         # "../dump/py-java/humaneval-py-java-PTremove-MTexplain-lbl-4shot-completion_trial2",
    #         # "../dump/py-java/humaneval-py-java-PTremove-MTexplain-lbl-4shot-completion_trial3",
    #         # "../dump/py-java/humaneval-py-java-PTremove-MTexplain-lbl-4shot-completion_trial4",
    #     ],
    #     base_dir="../dump/py-java/humaneval-py-java-PTremove-completion"
    # )

    # assert_no_tgt_lang_specific_explanation(
    #     "../dump/py-java/humaneval-py-java-PTremove-MTexplain-completion", "java"
    # )

    # collect_program_difficulty_level("rkt", "../dump")

    # print_non_trivial_problems("../dump/py-java/humaneval-py-java-PTremove-MTexplain20RR-completion")
    # print_intermediate_length("../dump/py-java/humaneval-py-java-PTremove-MTpivot-gold-php-completion")
    # print_all_tgt_specific_exp(4)