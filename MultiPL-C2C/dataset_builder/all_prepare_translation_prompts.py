"""
Run this script to call prepare_prompts_json.py for every experiment combination
in libexamples.
"""
import argparse
import os
import pdb
import shutil
import subprocess
from typing import *
from libexperiments import LANGS
from tmp.modify_completion_files import remove_target_info_from_completion_files


LANG_PAIRS = [
    ("py", "js"),    # high res -> high res
    ("js", "java"),
    ("cpp", "py"),
    ("java", "cpp"),
    ("js", "rkt"),    # high res -> low res
    ("py", "d"),
    ("cpp", "lua"),
    ("java", "jl"),
    ("lua", "py"),    # low res -> high res
    ("rkt", "java"),
    ("jl", "js"),
    ("d", "cpp"),
    ("lua", "rkt"),    # low res -> low res
    ("rkt", "jl"),
    ("d", "lua"),
    ("jl", "d"),
]

PYTHON_X_VARIATIONS = [
    {"shots": 0, "multiturn_prompt": "single"},  # baseline 0 shot
    {"shots": 0, "multiturn_prompt": "explain", "multi_view_dirs": "java,-completion-manually-remove-java"},  # explain 0 shot, java explanation
    {"shots": 0, "multiturn_prompt": "explain-lbl", "multi_view_dirs": "java,-completion-manually-remove-java"},  # explain-lbl 0 shot, java explanation
    {"shots": 0, "multiturn_prompt": "explain-lbl-simp", "multi_view_dirs": "java,-completion-manually-remove-java"},  # explain-lbl-simp 0 shot, java explanation
    {"shots": 4, "multiturn_prompt": "single"},  # baseline 4 shot
    {"shots": 4, "multiturn_prompt": "explain", "multi_view_dirs": "java,-completion-manually-remove-java"},  # explain 4 shot, 0 shot java explain
    {"shots": 4, "multiturn_prompt": "explain-lbl", "multi_view_dirs": "java,-4shot-completion"},  # explain-lbl 4 shot, 4 shot java explain-lbl
    {"shots": 4, "multiturn_prompt": "explain-lbl-simp", "multi_view_dirs": "java,-4shot-completion"},  # explain-lbl 4 shot, 4 shot java explain-lbl-simp (heuristic selected)
    # 2 trials below uses heuristics for explanation selection
    {"shots": 4, "multiturn_prompt": "explain", "multi_view_dirs": "java,-agg20_RRheuristic_coder_reviewer-shots-1_codegen2-16B_alpha0.7"},  # heuristically selected 0-shot explain 4 shot
    {"shots": 4, "multiturn_prompt": "explain-lbl-simp", "multi_view_dirs": "java,20RR-4shot-completion-agg20_RRheuristic_code_count_fragments"}  # explain-lbl 4 shot, 4 shot java explain-lbl-simp (heuristic selected)
]

X_X_VARIATIONS = [
    {"shots": 0, "multiturn_prompt": "single"},  # baseline 0 shot
    {"shots": 0, "multiturn_prompt": "explain"},  # explain 0 shot
    {"shots": 0, "multiturn_prompt": "explain-lbl"},  # explain-lbl 0 shot
    {"shots": 0, "multiturn_prompt": "explain-lbl-simp"},  # explain-lbl-simp 0 shot
    {"shots": 4, "multiturn_prompt": "single"},  # baseline 4 shot
    {"shots": 4, "multiturn_prompt": "explain", "multi_view_dirs": "java,-completion"},  # explain 4 shot, 0 shot self explain
    {"shots": 4, "multiturn_prompt": "explain-lbl"},  # explain-lbl 4 shot
    {"shots": 4, "multiturn_prompt": "explain-lbl-simp"}  # explain-lbl 4 shot
]

ALL_VARIATIONS = [
    {"shots": 0, "multiturn_prompt": "single"},  # baseline 0 shot
    {"shots": 0, "multiturn_prompt": "explain"},  # explain 0 shot
]

def multi_view_dirs(args, python_x: bool=True, exist_lang: Optional[str]=None, model: Optional[str]=None):
    multi_view_str = args.get('multi_view_dirs')
    if multi_view_str is None:
        return None
    if exist_lang is not None:
        multi_view_str = multi_view_str.replace("java", exist_lang)

    if "," in multi_view_str:
        intermediate_lang, postfix = multi_view_str.split(",")
    else:
        intermediate_lang, postfix = multi_view_str, ""

    model = "_"+model if model is not None else ""
    out = [
        f"../dump{model}/{args['src_lang']}-{intermediate_lang}/humaneval-{args['src_lang']}-{intermediate_lang}-PTremove-MT{args['multiturn_prompt']}{postfix}"
    ]
    return ",".join(out)


def few_shot_file(args):
    few_shot_mt_str = '_MT' + args['multiturn_prompt'] if args['shots'] > 0 and args['multiturn_prompt'] != "single" else ""
    few_shot_file = f"../few_shot_prompts/{args['tgt_lang']}/{args['src_lang']}-{args['tgt_lang']}_translate{few_shot_mt_str}.txt"
    return few_shot_file


def prepare_direction(tgt_lang: str, src_lang: str="py", python_x: bool=True, exist_lang: Optional[str]=None, model: Optional[str]=None, all_dir=False):
    p = "remove"
    sp = "keep"
    o = "../datasets/originals"

    for variation in ALL_VARIATIONS if all_dir else PYTHON_X_VARIATIONS if python_x else X_X_VARIATIONS:
        variation.update({
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        })
        mvd = multi_view_dirs(variation, python_x, exist_lang, model)
        fsf = few_shot_file(variation)
        kwargs_str = " ".join(f"--{k.replace('_','-')} {v}" for k, v in variation.items() if k not in
                          {"tgt_lang", "multi_view_dirs", "few_shot_file"})
        cmd = f"python3 prepare_prompts_json.py --lang humaneval_to_{tgt_lang}.py" + \
         f" --prompt-terminology {p} --originals {o} --source-program {sp} " \
         f" {kwargs_str}"
        cmd += f"--few-shot-file {fsf}" if variation["shots"] > 0 else ""
        cmd += f" --multi-view-dirs {mvd}" if mvd is not None else ""
        cmd += f" --output ../translation_prompts_{model}" if model is not None else ""
        # when cached explanations are not yet created (completion for that trial needs to happen first)
        if mvd and not os.path.exists(mvd):
            # if cache dir exists w/ completion, but still contain target lang specific info,
            if python_x and os.path.exists(mvd.replace(f"-manually-remove-{exist_lang}", "")):
                shutil.copytree(mvd.replace(f"-manually-remove-{exist_lang}", ""), mvd)
                remove_target_info_from_completion_files(mvd)
            else:
                print(f"Skipping {str(variation)} because cache trial {mvd} is not run yet. "
                      f"Please perform completion on the trial first before creating.")
                continue

        result = subprocess.run(cmd, shell=True, encoding="utf-8")

        # for x-x 4-shot exp trial, we name them to exp(0shot)-4shot for better documentation
        if not python_x and mvd is not None:
            four_shot_dir = mvd.replace("dump_", "translation_prompts_").replace("explain-", "explain-4shot-") + ".json"
            new_dir = four_shot_dir.replace("explain-", "explain(0shot)-")
            os.rename(four_shot_dir, new_dir)

        if result.returncode != 0:
            exit(1)


def prepare_python_xx(langs: Optional[List[str]]=None, exist_lang="java", model=None):
    for tgt_lang in LANGS if langs is None else langs:
        if tgt_lang not in {exist_lang, "py"}:
            prepare_direction(tgt_lang, exist_lang=exist_lang, model=model)


def prepare_xx_xx(pairs: Optional[List[Tuple[str, str]]]=None, model=None):
    for src_lang, tgt_lang in LANG_PAIRS if pairs is None else pairs:
        prepare_direction(tgt_lang, src_lang, python_x=False, exist_lang=tgt_lang, model=model)


def prepare_all_directions(model=None):
    for src_lang in LANGS:
        for tgt_lang in LANGS:
            if src_lang != tgt_lang:
                prepare_direction(tgt_lang, src_lang, exist_lang=None, model=model, all_dir=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=str, help="Set of experiment prompts to generate. Options: [py_x, x_x, all]")
    parser.add_argument("--model", type=str, default=None, help="Model name. Used to postfix completion and output directories")
    args = parser.parse_args()

    ##### X-X direction
    # To generate all directions for x-x for CodeGen2-1b model
    # prepare_xx_xx(model="codegen21b")
    # to generate selective pairs of x-x direction
    # prepare_xx_xx([("py", "js")], model="codegen21b")
    if args.trial == "x_x":
        prepare_xx_xx(model=args.model)

    ##### Python-X direction
    # if you want to re-use explanations from one single direction
    # for all other directions (like we did with Python-Java), you
    # first need to generate Python-Java
    # prepare_python_xx(["java"], exist_lang=None, model="codegen21b")

    # then you should run completions on Python-Java so model generated
    # explanations are added to each problem prompts in dump directory
    # then, run the rest of the directions using cached explanation
    # prepare_python_xx(exist_lang="java", model="codegen21b")

    # or if you want to generate selective pairs of Python-x direction
    # prepare_python_xx(["lua"], exist_lang="java", model="codegen21b")

    # However, if you do not care about caching and using the same
    # explanations across all directions, run the following
    # prepare_python_xx(exist_lang=None, model="codegen21b")
    if args.trial == "py_x":
        prepare_python_xx(exist_lang=None, model=args.model)

    if args.trial == "all":
        prepare_all_directions(args.model)


if __name__ == "__main__":
    main()


