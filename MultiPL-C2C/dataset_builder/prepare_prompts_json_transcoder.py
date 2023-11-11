"""
Similar as prepare_prompts_json.py, but for transcoder dataset (specifically C++ - Python)
"""

import argparse
import os.path
import sys

import pandas as pd
from typing import List, Optional

from codegen_sources.model.src.utils import TREE_SITTER_ROOT
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor
from dataset_builder.generic_translator import add_few_shot, modify_translation_prompt
from dataset_builder.utils import bool_flag, get_multi_vew_acronym, CANONICAL2SHORT, cap, \
    SHORT2CANONICAL
from generic_translator import list_originals, translate_prompt_and_tests, get_stop_from_translator
from pathlib import Path
import json

from translate_async import extract_gold_signature

lang2ext={
    "java": "java",
    "cpp": "cpp",
    "python": "py"
}

CPP_TO_PY_TYPE_MAP={
    "void": "None",
    "int": "int",
    "short": "int",
    "long": "int",
    "signed": "int",
    "unsigned": "int",
    "bool": "bool",
    "char": "str",
    "float": "float",
    "double": "float",
    "string": "str",
}

array_map1 = {f"{basic_type}[]": f"List[{py_type}]" for basic_type, py_type in CPP_TO_PY_TYPE_MAP.items()}
array_map2 = {f"{basic_type}*": f"List[{py_type}]" for basic_type, py_type in CPP_TO_PY_TYPE_MAP.items()}
CPP_TO_PY_TYPE_MAP.update(array_map1)
CPP_TO_PY_TYPE_MAP.update(array_map2)
CPP_TO_PY_TYPE_MAP["char*"] = "str"


def cpp2py(cpp_type: str):
    cpp_type = cpp_type.replace("&", "").replace("ll", "int").strip()
    if "vector" in cpp_type:
        depth = cpp_type.count("vector")
        cpp_type = cpp_type.replace("vector", "").replace("<", "").replace(">", "")
        cpp_type += "[]"*depth
    if cpp_type.replace(" ", "") in CPP_TO_PY_TYPE_MAP:
        return CPP_TO_PY_TYPE_MAP[cpp_type.replace(" ", "")]
    else:
        # nested list maybe
        depth = cpp_type.count("[")
        if depth > 1:
            out = CPP_TO_PY_TYPE_MAP[cpp_type.split()[0]]
            for _ in range(depth):
                out = f"List[{out}]"
            return out
        elif len(cpp_type.split()) > 1:  # multiple modifier to basic type(ex: unsigned long long)
            if cpp_type.split()[-1] in CPP_TO_PY_TYPE_MAP:
                return CPP_TO_PY_TYPE_MAP[cpp_type.split()[-1]]
            elif "=" in cpp_type:  # default values
                return cpp2py(" ".join(cpp_type.split()[:-2]))
        raise NotImplementedError()


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--src_lang", type=str, required=False, default="cpp", help="Language to translate from"
    )
    args.add_argument(
        "--tgt_lang", type=str, required=False, default="python", help="Language to translate to"
    )
    args.add_argument(
        "--input_program_tok_dir", type=str, required=False, default="../datasets/transcoder/transcoder_test_set",
        help="Folder containing .tok files of valid and test split programs"
    )
    args.add_argument(
        "--input_program_test_dir", type=str, required=False, default="../datasets/transcoder/transcoder_evaluation_gfg_fixed",
        help="Folder containing test files (which includes gold programs as well) of valid and test split programs"
    )
    args.add_argument(
        "--output", type=str, required=False, default="../translation_prompts/cpp-python/transcoder.json", help="Target JSON file"
    )
    args.add_argument(
        "--target-signature",
        type=str,
        default="keep",
        help="What to do with target program signature: remove (harder, not fully defined), keep (easier)"
    )
    args.add_argument(
        "--few_shot_file", type=str, required=False, default=None, help="few shot JSON file"
    )
    args.add_argument(
        "--shots", type=int, required=False, default=1,
        help="how many shots to use"
    )
    args.add_argument(
        "--multiturn-prompt",
        type=str,
        default="single",
        help="what type of multiturn prompt to generate: single (single turn), steps (generate NL steps, then "
             "translate), steps-cot (generate NL steps, translate each step w/ chain of thoughts provided w/ signature,"
             "combine w/ signature."
    )
    args.add_argument(
        "--prompt-type",
        type=str,
        default="completion",
        help="completion or chat (completion)"
    )
    args.add_argument("--split", type=str, default="valid")

    args = args.parse_args()
    return args


def keep_any_test_exist_ids(test_dir: str, in_data: List[str], tgt_lang: Optional[str]=None):
    ids = [l.split(" | ")[0] for l in in_data]
    tokenized_codes = [in_data[i].replace(f"{ids[i]} | ", "") for i in range(len(in_data))]

    keep_ids = []
    keep_codes = []
    tgt_langs = ["java", "python", "cpp"] if tgt_lang is None else [tgt_lang]
    for i, code in zip(ids, tokenized_codes):
        if any([os.path.isfile(f"{test_dir}/{l}/{i}.{lang2ext[l]}") for l in tgt_langs]):
            keep_ids.append(i)
            keep_codes.append(code)
    return keep_ids, keep_codes


def filter_with_existing_tests(in_data, test_dir, langs):
    common_ids = set([f.replace(f'.{langs[0]}', "") for f in os.listdir(f"{test_dir}/{langs[0]}")])
    for lang in langs[1:]:
        common_ids = common_ids.intersection([f.replace(f'.{lang}', "") for f in os.listdir(f"{test_dir}/{SHORT2CANONICAL[lang]}")])

    filt_data = []
    for data_line in in_data:
        if data_line.split("|")[0].strip() in common_ids:
            filt_data.append(data_line)
    return filt_data



def get_translation_candidates(in_dir: str, test_dir: str, split: str, src_lang: str, tgt_lang:str):
    lang_processor = LangProcessor.processors[src_lang](root_folder=TREE_SITTER_ROOT)
    if split == "combine" or split == "eval":
        in_data = [l.strip() for l in open(f"{in_dir}/transcoder_valid.{src_lang}.tok")]
        in_data.extend([l.strip() for l in open(f"{in_dir}/transcoder_test.{src_lang}.tok")])
        if split == "eval":  # weird split that PaLM and self-debug paper used for evaluation
            in_data = filter_with_existing_tests(in_data, test_dir, [src_lang, tgt_lang])
    else:
        in_data = [l.strip() for l in open(f"{in_dir}/transcoder_{split}.{src_lang}.tok")]
    ids, tokenized_codes = keep_any_test_exist_ids(test_dir, in_data, SHORT2CANONICAL[tgt_lang])
    codes = [lang_processor.detokenize_code(c) for c in tokenized_codes]
    return zip(ids, codes)


def ensure_tokenization(function):
    # some manual way of correcting un-tokenized input function string to avoid weird error output
    if " ( " not in function:
        function = function.replace("(", " ( ", 1)
    if " ) " not in function:
        function = function.replace(")", " ) ", 1)
    return function


def add_type_cpp_to_python_signature(py_sig, cpp_sig):
    cpp_processor = LangProcessor.processors["cpp"](root_folder=TREE_SITTER_ROOT)
    cpp_fname = cpp_processor.get_function_name(cpp_sig)
    cpp_inputs = cpp_processor.extract_arguments(cpp_sig)
    cpp_output_type = cpp_sig[:cpp_sig.find(cpp_fname)].replace("public", "").replace(":","").strip()
    py_processor = LangProcessor.processors["python"](root_folder=TREE_SITTER_ROOT)

    py_sig = ensure_tokenization(py_sig)
    py_inputs = py_processor.extract_arguments(py_sig)
    py_output_type = cpp2py(cpp_output_type)
    assert len(cpp_inputs[0]) == len(py_inputs[0]) and len(cpp_inputs[1]) == len(py_inputs[1])
    py_typed_inputs = []
    for i, (py_input, cpp_input_type) in enumerate(zip(py_inputs[1], cpp_inputs[0])):
        if "=" in py_input or "=" in cpp_input_type:
            # if there is default value, transcoder parser doesn't work all the time, need messy parsing
            var_name, var_default = None, None
            for var_str in [py_input, cpp_input_type]:
                if "=" in var_str:
                    if var_name is None:
                        var_name = var_str.split("=")[0].split()[-1]
                    right_side = var_str.split("=")[1].strip()
                    if len(right_side) > 0:
                        var_default = right_side
            if var_default is None and cpp_inputs[1][i].strip() != "":
                var_default = cpp_inputs[1][i].strip()
            if var_default is not None:
                py_typed_input = f"{var_name}: {cpp2py(cpp_input_type)}={var_default}"
            else:
                py_typed_input = f"{var_name}: {cpp2py(cpp_input_type)}"
        else:
            py_typed_input = f"{py_input.strip()}: {cpp2py(cpp_input_type)}"
        py_typed_inputs.append(py_typed_input)
    py_typed_inputs = ", ".join(py_typed_inputs)
    typed_py_sig = py_sig[:py_sig.find("(")+1] + py_typed_inputs + py_sig[py_sig.find(")"):]
    typed_py_sig = typed_py_sig.strip()[:-1] + f" -> {py_output_type}:"
    if any([w in py_typed_inputs for w in ["List[", "Dict["]]):
        typed_py_sig = "from typing import *\n\n" + typed_py_sig
    return typed_py_sig


def extract_transcoder_prompt_and_tests(pid, code, test_dir, target_signature, src_lang, tgt_lang, few_shots, shots):
    if tgt_lang == "py":
        test_path = f"{test_dir}/python/{pid}.{tgt_lang}"
    else:
        test_path = f"{test_dir}/{tgt_lang}/{pid}.{tgt_lang}"

    # In reality, we will evaluate using transcoder eval, so these tests varaibles don't
    # really matter, what matters is that the prompt is correct
    if target_signature == "keep":
        tgt_sig = extract_gold_signature(test_path, src_code=code)
    elif target_signature == "typed":
        tgt_sig = extract_gold_signature(test_path, src_code=code)
        if tgt_sig != "":
            tgt_sig = add_type_cpp_to_python_signature(tgt_sig, code.strip().split("\n")[0])
    else:
        if tgt_lang == "py":
            tgt_sig = "def"
        else:
            raise NotImplementedError()
    if tgt_sig == "":
        return None

    prompt_str = f"### {cap(SHORT2CANONICAL[src_lang])} version\n\n{code.strip()}\n\n" \
                 f"### {cap(SHORT2CANONICAL[tgt_lang])} version\n\n{tgt_sig}"

    translated_prompt = add_few_shot(prompt_str, few_shots, shots, tgt_lang)
    if translated_prompt is None:  # when the problem is in few shot
        return None

    return translated_prompt, ""


def main(args):
    translator = __import__(f"humaneval_to_{args.tgt_lang}").Translator()

    if args.src_lang not in ["cpp", "py", "java"]:
        print(f"Invalid source lang option: {args.src_lang}")
        sys.exit(1)

    if args.tgt_lang not in ["cpp", "py", "java"]:
        print(f"Invalid target lang option: {args.src_lang}")
        sys.exit(1)

    if args.target_signature not in ["keep", "remove", "typed"]:
        print(f"Invalid target signature option: {args.src_lang}")
        sys.exit(1)

    if args.few_shot_file is not None:
        if args.few_shot_file.endswith(".csv"):
            few_shot_data = pd.read_csv(args.few_shot_file)
        elif args.few_shot_file.endswith(".jsonl"):
            few_shot_data = [json.loads(l) for l in open(args.few_shot_file)]
        else:
            assert args.few_shot_file.endswith(".txt") and args.prompt_type == "completion"
            few_shot_data = open(args.few_shot_file).read()

    else:
        few_shot_data, args.few_shot_file = [], ""

    translation_candidates = get_translation_candidates(args.input_program_tok_dir, args.input_program_test_dir, args.split, args.src_lang, args.tgt_lang)

    os.makedirs(Path(args.output).parent, exist_ok=True)
    results = [ ]
    for original in translation_candidates:
        original_name, original_program = original
        print(f"Processing {original_name}...")

        few_shot = few_shot_data if isinstance(few_shot_data, list) or isinstance(few_shot_data, str) else \
            few_shot_data.loc[original_name]
        result = extract_transcoder_prompt_and_tests(
            original_name, original_program, args.input_program_test_dir, args.target_signature, args.src_lang, args.tgt_lang,
            [] if "MT" in args.few_shot_file else few_shot, args.shots
        )
        if result is None:
            print(f"Skipping {original_name}")
            continue

        (prompt, tests) = result
        problem = {
            "name": original_name,
            "language": args.tgt_lang,
            "prompt": prompt,
            "tests": tests,
            "stop_tokens": get_stop_from_translator(translator),  # TODO may only work for python
        }
        problem = modify_translation_prompt(problem, args.multiturn_prompt,
                                            few_shot if "MT" in args.few_shot_file else [], args.shots,
                                            src_lang=cap(SHORT2CANONICAL[args.src_lang]),
                                            multi_view_files=[])
        if problem is None:
            print(f"skipping {original_name}")
            continue

        results.append(problem)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = get_args()
    args.src_lang = "cpp"
    args.tgt_lang = "py"
    args.few_shot_file = f"../few_shot_prompts/{args.tgt_lang}/{args.src_lang}-{args.tgt_lang}_translate.txt"
    args.shots = 0
    args.target_signature = "typed"
    args.multiturn_prompt = "explain"
    args.prompt_type = "completion"
    args.split="eval"
    args.input_program_tok_dir = "../../decompose-and-translate/data/transcoder_test_set"
    args.input_program_test_dir = "../../CodeGenMirror/data/transcoder_evaluation_gfg"

    ds_name = args.input_program_tok_dir.split('/')[-1].replace("_test_set", "")+"_"+args.split
    args.output = f"../translation_prompts/{args.src_lang}-{args.tgt_lang}/{ds_name}-{args.src_lang}-{args.tgt_lang}"
    if args.target_signature != "remove":
        args.output += f"-TS{args.target_signature}"
    if args.multiturn_prompt != "single":
        args.output += f"-MT{args.multiturn_prompt}"
        if args.multiturn_prompt == "multi-view":
            args.output += "_" + get_multi_vew_acronym(args.multi_view_dirs)
    if args.few_shot_file is not None and args.shots > 0:
        args.output += f"-{args.shots}shot"
    if args.prompt_type == "completion":
        args.output += f"-completion"
    args.output += ".json"
    main(args)
