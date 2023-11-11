import json
import os
from functools import partial
from typing import *
import regex as re

import numpy as np
from tqdm import tqdm

from codegen_sources.model.src.utils import TREE_SITTER_ROOT
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor
from dataset_builder.utils import get_source_code_from_prompt, replace_token
from retrieval.retrieve_utils import retrieve_few_shot_indices_only

RANDOM_PROGRAM_MAPPING = {}
RETRIEVAL_PROGRAM_MAPPING = {}


def get_formal_counterfactual_pivots(problem: Dict) -> str:
    """
    For each formal language as pivot experiment, find a generated program that is wrong,
    and by comparing using correct vs. wrong formal pivot we hope to observe how
    sensitive the models are to the correctness of formal language as latent guidance
    """
    code, pivot = get_code_and_exp(problem)
    # MOVING THIS TO prepare_prompts_json as a multi_turn option
    pass


def swap_explanation_sentence(problem: Dict) -> str:
    """
    swap sentence, or segments of line explanation around to observe whether the
    order of the explanation impact accuracy
    """
    code, exp = get_code_and_exp(problem)
    delimiter = "\n\n" if "line by line" in problem["translation_prompt"] else ". "
    if exp.endswith(".") and delimiter == ". ":
        exp = exp[:-1]

    exp_segments = exp.split(delimiter)
    np.random.shuffle(exp_segments)
    swapped_exp = delimiter.join(exp_segments)
    if delimiter==". ":
        swapped_exp += "."

    return swapped_exp


def delete_explanation_sentence(problem: Dict, frac: float) -> str:
    """
    delete sentence, or segments of line explanation around to observe whether the
    order of the explanation impact accuracy
    """
    code, exp = get_code_and_exp(problem)
    delimiter = "\n\n" if "line by line" in problem["translation_prompt"] else ". "
    if exp.endswith(".") and delimiter == ". ":
        exp = exp[:-1]
    exp_segments = exp.split(delimiter)
    if len(exp_segments) > 1:
        del_indices = set(np.random.choice(len(exp_segments), int(np.ceil(frac * len(exp_segments))), replace=False))
        left_segments = [s for i, s in enumerate(exp_segments) if i not in del_indices]
        alt_exp = delimiter.join(left_segments)
    else:
        alt_exp = exp
    if delimiter==". ":
        alt_exp += "."
    return alt_exp


def get_explain_type_from_prompt(prompt: str):
    return "explain-lbl-simp" if "simplify it and explain" in prompt \
            else "explain-lbl" if "line by line" in prompt else "explain"


def obfuscated_code_explanation(problem) -> str:
    """
    obfusecate code (if python use TransCoder), then generate explanation
    """
    explain_type = get_explain_type_from_prompt(problem["translation_prompt"])
    obf_dump_dir = f"../dump/py-{problem['language']}/humaneval-py-{problem['language']}-PTremove-MT{explain_type}-obf-completion"
    if problem["language"] == "java":
        obf_dump_dir += "-manually-remove-java"
    obf_dump_path = f"{obf_dump_dir}/{problem['name']}.json"
    with open(obf_dump_path) as f:
        obf_data = json.load(f)
    obf_exp = get_source_code_from_prompt(obf_data["translation_prompt"])
    return obf_exp


def obfuscated_variable_in_explanation(problem) -> str:
    """
    extract variable reference using code fragment regex ``, then replace them with
    a permutations of other variables in the code
    """
    code, exp = get_code_and_exp(problem)
    code_frags = re.findall(r"`[^`]*`", exp)
    all_vars = {}
    for code_frag in code_frags:
        vars = re.findall(r"[^\W0-9]\w*", code_frag)
        all_vars.update(vars)
    all_vars1 = list(all_vars)
    all_vars2 = all_vars1.copy()
    np.random.shuffle(all_vars2)
    vars_map = {v1: v2 for v1, v2 in zip(all_vars1, all_vars2)}
    vars_order = sorted(all_vars1, lambda x: len(x), reverse=True)
    for code_frag in code_frags:
        # can be cases where it falsely replace sub-expression with variable name just
        # because that variable name is short or something, but ignore for now since we
        # want to ablate and see decrease in performance
        new_code_frag = code_frag
        for v in vars_order:
            new_code_frag = new_code_frag.replace(v, vars_map[v])
        exp.replace(code_frag, new_code_frag)

    return exp


def delete_explanation_word(problem: Dict, frac: float) -> str:
    """
    delete random works, try to keep \n and indent as separate tokens
    """
    code, exp = get_code_and_exp(problem)
    new_exp = replace_token(exp.replace("\n"," NEWLINE "), {"    ": " INDENT "})
    words = new_exp.split(" ")
    del_indices = set(np.random.choice(len(words), max(int(frac*len(words)), 1), replace=False))
    left_words = [w for i, w in enumerate(words) if i not in del_indices]
    new_exp = " ".join(left_words)
    new_exp = replace_token(new_exp, {"NEWLINE": "\n", "INDENT": "    "}).replace("\n ", "\n").replace(" \n", "\n")
    return new_exp


def get_other_program_explanation(problem: Dict, dump_dir: str) -> str:
    global RANDOM_PROGRAM_MAPPING
    if len(RANDOM_PROGRAM_MAPPING) == 0:
        all_problem_names = list(set([f.replace(".json", "") for f in os.listdir(dump_dir) if not f.endswith(".results.json")]))
        all_problem_names_shuff = all_problem_names.copy()
        np.random.shuffle(all_problem_names_shuff)
        while any([all_problem_names[i]==all_problem_names_shuff[i] for i in range(len(all_problem_names))]):
            np.random.shuffle(all_problem_names_shuff)
        RANDOM_PROGRAM_MAPPING = {k:v for k, v in zip(all_problem_names, all_problem_names_shuff)}

    if problem["name"] not in RANDOM_PROGRAM_MAPPING:
        RANDOM_PROGRAM_MAPPING[problem["name"]] = np.random.choice(list(RANDOM_PROGRAM_MAPPING.keys()))
    random_name = RANDOM_PROGRAM_MAPPING[problem["name"]]
    dump_path = f"{dump_dir}/{random_name}.json"
    with open(dump_path) as f:
        random_problem = json.load(f)
    random_code, random_exp = get_code_and_exp(random_problem)
    return random_exp


def get_other_program_explanation_retrieval(problem: Dict, method: str, dump_dir: str) -> str:
    global RETRIEVAL_PROGRAM_MAPPING
    if problem["name"] not in RETRIEVAL_PROGRAM_MAPPING:
        # TODO not hardcode python
        RETRIEVAL_PROGRAM_MAPPING, _, _ = retrieve_few_shot_indices_only(dump_dir, dump_dir, "python", "python", method, 2, True)

    retrieval_name = RETRIEVAL_PROGRAM_MAPPING[problem["name"]][1]  # first one is always query program itself
    dump_path = f"{dump_dir}/{retrieval_name}.json"
    with open(dump_path) as f:
        random_problem = json.load(f)
    random_code, random_exp = get_code_and_exp(random_problem)
    return random_exp


def get_code_and_exp(problem: Dict) -> Tuple[str, str]:
    src_code = get_source_code_from_prompt(problem["translation_prompt"], -3)
    exp = get_source_code_from_prompt(problem["translation_prompt"])
    return src_code.strip(), exp.strip()


def get_method_factory(dump_dir: str, method: str) -> Callable:
    match method.split("-"):
        case ["swap", "sent"]:
            return swap_explanation_sentence
        case ["obf", "exp"]:
            return obfuscated_code_explanation
        case ["obf", "var"]:
            return obfuscated_variable_in_explanation
        case ["del", "sent", frac]:
            frac = float(frac)
            return partial(delete_explanation_sentence, frac=frac)
        case ["del", "word", frac]:
            frac = float(frac)
            return partial(delete_explanation_word, frac=frac)
        case ["other", "exp"]:
            return partial(get_other_program_explanation, dump_dir=dump_dir)
        case ["other", "exp", retrieval_method]:
            return partial(get_other_program_explanation_retrieval, method=retrieval_method, dump_dir=dump_dir)
        case ["no", "exp"]:
            return lambda x: ""


def generate_alternative_explanations(dump_dir: str, method: str):
    out_dump_dir = f"{dump_dir}-ALT{method}"
    os.makedirs(out_dump_dir, exist_ok=True)
    out_prompt_path = f"{dump_dir.replace('dump','translation_prompts')}-ALT{method}.json"
    problems = []
    alt_cnt = 0
    files = [f for f in os.listdir(dump_dir) if not f.endswith(".results.json")]
    total = len(files)
    for f in tqdm(files):
        with open(f"{dump_dir}/{f}") as f:
            problem = json.load(f)

        get_method = get_method_factory(dump_dir, method)
        alt_exp = get_method(problem)
        if alt_exp is None:
            continue
        _, exp = get_code_and_exp(problem)
        problem["translation_prompt"] = problem["translation_prompt"].replace(exp.strip(), alt_exp.strip())
        if exp.strip() != alt_exp.strip():
            alt_cnt += 1
        else:
            out_dump_path = f"{out_dump_dir}/{problem['name']}.json"
            with open(out_dump_path, "w") as f:
                json.dump(problem, f)  # intentionally no indent so we know which ones were copied
        del problem["completions"]
        problems.append(problem)

    print(f"Dump dir: {dump_dir}\nMethod={method}, Total of {alt_cnt}/{total} ({alt_cnt/total*100:.1f}%) explanation altered")
    with open(out_prompt_path, "w") as f:
        json.dump(problems, f, indent=2)


if __name__ == "__main__":
    np.random.seed(42)
    for lang in "java php swift rkt".split():  #java php swift rkt
        dump_dirs = [
            f"../dump/py-{lang}/humaneval-py-{lang}-PTremove-MTexplain-completion-manually-remove-java",
            f"../dump/py-{lang}/humaneval-py-{lang}-PTremove-MTexplain-lbl-completion-manually-remove-java",
        ] if lang == "java" else [
            f"../dump/py-{lang}/humaneval-py-{lang}-PTremove-MTexplain-completion",
            f"../dump/py-{lang}/humaneval-py-{lang}-PTremove-MTexplain-lbl-completion",
        ]
        for dump_dir in dump_dirs:
            for method in [
                "del-word-0.25",
                "del-word-0.5",
                "swap-sent",
                "del-sent-0.25",
                "del-sent-0.5",
                "obf-exp",
                "other-exp",
                "other-exp-bm25",
                "no-exp"
            ]:
                generate_alternative_explanations(dump_dir, method)
