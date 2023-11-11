from collections import defaultdict
from typing import *
import regex as re

from codegen_sources.model.src.utils import TREE_SITTER_ROOT
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor


def get_output_type_java(func_name, java_func):
    func_name_idx = (
        re.compile(r" (%s)\b" % re.escape(func_name)).search(java_func).start()
    )
    out_start = max(
        [
            i + 6
            for i in (
            java_func.find("public"),
            java_func.find("static"),
            java_func.find("private"),
            java_func.find("final"),
            0,
        )
            if i < func_name_idx
        ]
    )
    out_type = java_func[out_start:func_name_idx].strip()
    if not ">" in out_type and not "]" in out_type and " " in out_type:
        out_type = out_type.split(" ")[-1].strip()
    return out_type


def get_output_type_python(func_name:str, python_func: str) -> str:
    if "->" in python_func:
        start_idx = python_func.find("->", python_func.find(")")) + 2
        end_idx = python_func.find(":", start_idx)
        return python_func[start_idx: end_idx].strip()
    else:
        return ""


def get_output_type(function, func_name, lang):
    if lang == "java":
        ref_output_type = get_output_type_java(func_name, function)
    elif lang == "python":
        ref_output_type = get_output_type_python(func_name, function)
    else:
        raise NotImplementedError
    return ref_output_type


def clean_inputs(inputs: Tuple[List[str], List[str]]) -> Tuple[List[str], List[str]]:
    for i in range(len(inputs[0])):
        inputs[0][i] = inputs[0][i].replace("final ","").replace(" ", "")
    return inputs


def static_analysis(functions_list: List[str], ref: str, lang: str) -> Tuple[Dict, Dict]:
    # analyze whether function signatures are essentially the same (check function name, output type, input types)
    lang_processor = LangProcessor.processors[lang](root_folder=TREE_SITTER_ROOT)
    ref_tokens = " ".join(lang_processor.tokenize_code(ref))  # New to python, not checked with java
    ref_f_name = lang_processor.get_function_name(ref_tokens)
    ref_inputs = clean_inputs(lang_processor.extract_arguments(ref_tokens))
    ref_output_type = get_output_type(ref, ref_f_name, lang)
    type_translation = defaultdict(set)
    results = []
    for i, translation in enumerate(functions_list):
        try:
            translation_tok = " ".join(lang_processor.tokenize_code(translation))
            f_name = lang_processor.get_function_name(translation_tok)
            inputs = clean_inputs(lang_processor.extract_arguments(translation_tok))
            output_type = get_output_type(translation, f_name, lang)
            type_translation[ref_output_type].add(output_type)
            for i, ref_input in enumerate(ref_inputs[0]):
                if i < len(inputs[0]):
                    type_translation[ref_input.strip()].add(inputs[0][i].strip())
            result = {
                "match_f_name": f_name == ref_f_name,
                "input_len_diff":  len(ref_inputs[0]) - len(inputs[0]),
                "input_type_diff": sum([ref_inputs[0][i].strip().lower()!=inputs[0][i].strip().lower() for i in range(min(len(ref_inputs[0]),len(inputs[0])))]),
                "result_in_input": any([inputs[1][i].lower() == "result" for i in range(len(inputs[0]))]),
                "match_output_type": output_type == ref_output_type
            }
        except Exception as e:
            print(f"exception during static analysis! {e}")
            result = {
                "match_f_name": ref_f_name in translation,
                "input_len_diff": 1,
                "input_type_diff": 1,
                "result_in_input": "result" in translation,
                "match_output_type": False
            }
        results.append(result)

    return results, type_translation


def analysis_by_method(result_output):
    # calculate nuanced failure modes:
    # how many failure caused by having "result" in signature
    # how many failure caused by having incorrect number of inputs
    # how many failures caused by having incorrect types (with same of input lengths)
    # how many failures caused by having incorrect output type
    # how many failures has nothing to do with signatures
    # as long as one beam of the translation causes failure, it is counted for that reason.
    result_stats={
        "failure_by_f_name": 0,
        "failure_by_result": 0,
        "failure_by_input_len": 0,
        "failure_by_input_types": 0,
        "failure_by_output_type": 0,
        "failure_otherwise": 0
    }
    for id, output in result_output.items():
        failure = not any([o[0]=="success" for o in output])
        if not failure or (not isinstance(output[0][-1], dict)):
            continue
        if any([not o[-1]["match_f_name"] for o in output]):
            result_stats["failure_by_f_name"] += 1
        if any([o[-1]["result_in_input"] for o in output]):
            result_stats["failure_by_result"] += 1
        if any([o[-1]["input_len_diff"] != 0 for o in output]):
            result_stats["failure_by_input_len"] += 1
        if any([(o[-1]["input_len_diff"] == 0) and (o[-1]["input_type_diff"] != 0) for o in output]):
            result_stats["failure_by_input_types"] += 1
        if any([not o[-1]["match_output_type"] for o in output]):
            result_stats["failure_by_output_type"] += 1
        if any([(o[-1]["input_len_diff"] == 0) and (o[-1]["input_type_diff"] == 0)
                and o[-1]["match_f_name"] and not o[-1]["result_in_input"]
                and o[-1]["match_output_type"] for o in output]):
            result_stats["failure_otherwise"] += 1

    return result_stats


if __name__ == "__main__":
    pass