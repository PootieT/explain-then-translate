import os
import json
from pathlib import Path
from typing import *
import sys

from dataset_builder.utils import SHORT2CANONICAL, cap, CANONICAL2SHORT

sys.path.append(str(Path(__file__).parents[1].joinpath("dataset_builder")))

DEFAULT_FEW_SHOT_NAMES = [
    "HumanEval_107_even_odd_palindrome",
    "HumanEval_126_is_sorted",
    "HumanEval_1_separate_paren_groups.py",
    "HumanEval_88_sort_array",
]


def fuzzy_get_file_name(code):
    if "palindrome" in code.lower():
        return "HumanEval_107_even_odd_palindrome"
    elif "separate" in code.lower():
        return "HumanEval_1_separate_paren_groups"
    elif "sorted" in code.lower() and any([w in code.lower() for w in ["count_digit", "countdigit", "count-digit"]]):
        return "HumanEval_126_is_sorted"
    elif "array" in code.lower():
        return "HumanEval_88_sort_array"
    else:
        raise Exception("Do not recognize the function name")


def generate(example_path: str, gold_dir: str, langs: Optional[List[str]]=None, src_lang: str="py"):
    replace_src_from_python = False
    if not os.path.exists(example_path):
        if "MT" in example_path:
            raise FileNotFoundError("For few shot examples with intermediate steps, one needs to have an existing example few shot file")
        else:
            # if just baseline file, we can generate from py-xx few shot file, and replace python w/ src_lang
            example_path = example_path.replace(src_lang, "py")
            replace_src_from_python = True
    example_path = Path(example_path)
    lang_short = example_path.name.split("_")[0].split("-")[-1]
    lang = cap(SHORT2CANONICAL[lang_short])
    fs_dir = example_path.parent.parent
    with open(example_path) as f:
        template = f.read()

    if replace_src_from_python:
        template = replace_lang_and_code(gold_dir, "Python", template, src_lang)

    tgt_langs = langs if langs is not None else ["cpp", "cs", "d", "go", "js", "php", "pl", "rb", "rkt", "rs", "scala", "swift", "ts"]
    for tgt_lang_short in tgt_langs:
        if replace_src_from_python:
            name = str(example_path.name).replace("py", src_lang).replace(lang_short, tgt_lang_short)
        else:
            name = str(example_path.name).replace(lang_short, tgt_lang_short)
        tgt_path = fs_dir.joinpath(tgt_lang_short, name)
        if tgt_path.exists():
            continue

        print(f"generating few shot for {tgt_lang_short}...")

        # now loop through all tgt_lang section, go through and replace tgt lang code with whatever we can find in gold
        fs_string = replace_lang_and_code(gold_dir, lang, template, tgt_lang_short)

        with open(tgt_path, "w") as f:
            f.write(fs_string)


def replace_lang_and_code(gold_dir, lang, template, tgt_lang_short):
    gold_programs = json.load(open(f"{gold_dir}/{tgt_lang_short}.json"))
    tgt_translator = __import__(f"humaneval_to_{tgt_lang_short}").Translator()

    tgt_lang = cap(SHORT2CANONICAL[tgt_lang_short])
    if tgt_lang == lang:  # if the lang to replace is the same as original, return template
        return template

    fs_string = template.replace(f"{lang}?", f"{tgt_lang}?").replace(f"{lang} code", f"{tgt_lang} code").replace(f"{lang} program", f"{tgt_lang} program")
    while f"### {lang} version" in fs_string:
        start = fs_string.find(f"### {lang} version")
        code_start = start + len(f"### {lang} version")
        end = fs_string.find("### ", code_start)
        src_code = fs_string[code_start: end].strip()
        program_name = fuzzy_get_file_name(src_code)
        tgt_code = gold_programs.get(program_name, "")
        if tgt_code == "":
            print(f"tgt_lang={tgt_lang}, program={program_name} does not have existing gold")
        else:
            tgt_code = tgt_translator.remove_tests(tgt_code).strip()
        fs_string = fs_string.replace(src_code, tgt_code)
        fs_string = fs_string[:start] + f"### {tgt_lang} version" + fs_string[code_start:]
    return fs_string


if __name__ == "__main__":

    src_lang = "py"
    exist_tgt_lang = "cpp"
    for src_prompt in [
        f"../few_shot_prompts/{exist_tgt_lang}/{src_lang}-{exist_tgt_lang}_translate.txt",
        f"../few_shot_prompts/{exist_tgt_lang}/{src_lang}-{exist_tgt_lang}_translate_MTexplain.txt",
        f"../few_shot_prompts/{exist_tgt_lang}/{src_lang}-{exist_tgt_lang}_translate_MTexplain-lbl.txt",
        f"../few_shot_prompts/{exist_tgt_lang}/{src_lang}-{exist_tgt_lang}_translate_MTexplain-lbl-simp.txt"
    ]:
        generate(
            src_prompt,
            "../datasets/humaneval_multi",
            langs=["r"],
            src_lang=src_lang
        )