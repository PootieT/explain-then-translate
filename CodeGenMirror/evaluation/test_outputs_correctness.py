import os
import pprint
import re
import json
import pathlib
from collections import defaultdict, Counter
from typing import Optional, Dict, Union

import pandas as pd
from tqdm import tqdm

import argparse
from codegen_sources.model.src.evaluation.comp_acc_computation_modified import (
    submit_functions,
    submit_cobol_functions, TREE_SITTER_ROOT, submit_fortran_functions, submit_ibm_functions,
)


## example of results file: (function can also be a list of functions from multiple beams)
# {"ADD_1_TO_A_GIVEN_NUMBER": {
#          "lang": "python",
#          "function": "def f_gold ( x ) :\n    m = 1 ;\n    while ( x & m ) :\n        x = x ^ m\n        m <<= 1\n    x = x ^ m\n    return x\n"
#      }
# }
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor
from evaluation.utils import static_analysis, analysis_by_method


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_file",
        type=str,
        default="/home/tangpihai/Project/generate_candidate-and-translate/data/transcoder_evaluation_gfg/no_extract/python/python.json",
    )
    parser.add_argument(
        "--outfolder",
        type=str,
        default="/home/tangpihai/Project/generate_candidate-and-translate/dump/no_extract/decomposition_validation",
        help="output folder",
    )
    parser.add_argument(
        "--script_folder",
        type=str,
        default="/home/tangpihai/Project/CodeGenMirror/data/transcoder_evaluation_gfg",
        help="script folder",
    )
    parser.add_argument(
        "--roberta_mode",
        type=bool,
        default=False,
        help="Whether to use roberta mode, use True if input is not tokenized programs",
    )
    parser.add_argument(
        "--retry_mismatching_types",
        type=bool,
        default=False,
        help="Whether to retry mistyped function evaluation",
    )
    return parser


lang2ending = {
    "python": ".py",
    "cpp": ".cpp",
    "java": ".java",
    "cobol": ".cbl",
    "fortran": ".f90"
}


def get_ref_f_name(script_folder: Union[str, pd.DataFrame], id, lang) -> Optional[str]:
    ending = lang2ending[lang]
    if isinstance(script_folder, str):
        ref_file = script_folder + "/" + lang + "/" + id + ending
        if os.path.exists(ref_file):
            ref_code = open(ref_file).read()
        else:
            # raise Exception("No ref file found for {}".format(id))
            return None
    else:
        if id not in script_folder.index:
            return None
        ref_code = script_folder.loc[id]["code"]
    if lang == "java":
        res = re.search(f"{id}\.([\w]+)\(", ref_code)
        if res is not None:
            f_name = res.group(1)
        else:
            return "f_filled"
    else:
        return "f_filled"
    return f_name


def get_ref(script_folder: Union[str, pd.DataFrame], id, lang) -> Optional[str]:
    ending = lang2ending[lang]
    if isinstance(script_folder, str):
        ref_file = script_folder + "/" + lang + "/" + id + ending
        if os.path.exists(ref_file):
            ref_code = open(ref_file).read()
        else:
            # raise Exception("No ref file found for {}".format(id))
            return None
    else:
        if id not in script_folder.index:
            return None
        ref_code = script_folder.loc[id]["code"]
    if lang in lang2ending:
        ref_code = get_program_ref(ref_code, lang)
    else:
        raise NotImplementedError("Only python/cpp/java refs are supported")
    return ref_code

## regex, start at def and end at #TOFILL
LANG2REGEX = {
    "REMOVE_PYTHON_MAIN": re.compile(r"#TOFILL[\s\S]*"),
    "REMOVE_PYTHON_COMMENTS": re.compile(r"#.*"),
    "REMOVE_JAVA_MAIN": re.compile(r"//TOFILL[\s\S]*"),
    "REMOVE_JAVA_COMMENTS": re.compile(r"//.*"),
    "REMOVE_JAVA_IMPORTS": re.compile(r"import .*"),
    "REMOVE_CPP_MAIN": re.compile(r"//TOFILL[\s\S]*"),
    "REMOVE_CPP_COMMENTS": re.compile(r"//.*"),
}


def get_program_ref(ref_code: str, lang: str):
    if lang == "cobol":
        return ""

    ref_code = LANG2REGEX[f"REMOVE_{lang.upper()}_MAIN"].sub("", ref_code)
    ref_code = LANG2REGEX[f"REMOVE_{lang.upper()}_COMMENTS"].sub("", ref_code)
    if lang == "java": # and "junit" in ref_code:  # for evosuite tests (and seems valid/test set as well)
        ref_code = LANG2REGEX[f"REMOVE_{lang.upper()}_IMPORTS"].sub("", ref_code)
        if " class " in ref_code:
            ref_code = ref_code[ref_code.find("{")+1:]
    if "f_gold" not in ref_code:
        return ""
    ref_code = ref_code.replace("f_gold", "f_filled")
    return ref_code.strip()


def submit_generations(
    generation_results: Dict,
    script_folder: str,
    outfolder: str,
    roberta_mode: bool = False,
    retry_mismatching_types: bool = False,
    run_all_beams: bool = False,
    refactor_before: bool = True,
    criteria="",
):
    print(f"evaluating {outfolder} ...")
    os.makedirs(outfolder, exist_ok=True)
    output_results = {}
    all_results, all_statuses = [], []
    n_success, n_success_10, n_partial_success, n_partial_success_10, total = (
        0,
        0,
        0,
        0,
        0,
    )
    result_stats = {
        "success": 0,
        "failure": 0,
        "error": 0,
        "timeout": 0,
        "script_not_found": 0,
        "identical_gold": 0,
        "total": 0,
        "static_total": 0
    }
    type_translation = defaultdict(Counter)
    if os.path.isfile(script_folder) and script_folder.endswith(".csv"):
        script_df = pd.read_csv(script_folder, index_col="TARGET_CLASS")
    else:
        script_df = None

    with tqdm(generation_results.items()) as pbar:
        for id, result in generation_results.items():
            lang = result["lang"]
            result_stats["total"] += 1
            pbar.update(1)
            result_functions = result["function"] if isinstance(result["function"], list) else [result["function"]]
            ref = get_ref(script_df if script_df is not None else script_folder, id, lang if lang !="fortran" else "python")

            # if target unit test not found for translated program, skip
            # otherwise we are okay if script_df contains input/output column (ibm dataset)
            if ref is None:
                if script_df is None or (script_df is not None and "output" not in script_df.columns):
                    result_stats["script_not_found"] += 1
                    output_results[id] = ["error", "script_not_found"]
                    continue

            if lang.lower() == "cobol":
                list_of_fun_results, i = submit_cobol_functions(
                    functions_list=result_functions,
                    id=id,
                    outfolder=outfolder,
                    script_folder=script_df if script_df is not None else script_folder,
                    run_all_beams=run_all_beams,
                    refactor_before=refactor_before,
                )
            elif lang.lower() == "fortran":
                list_of_fun_results, i = submit_fortran_functions(
                    functions_list=result_functions,
                    id=id,
                    outfolder=outfolder,
                    script_folder=script_df if script_df is not None else script_folder,
                    run_all_beams=run_all_beams,
                )
            elif script_df is not None and "output" in script_df:
                list_of_fun_results, i = submit_ibm_functions(
                    functions_list=result_functions,
                    id=id,
                    lang=lang,
                    outfolder=outfolder,
                    test_input=script_df.loc[id]["input"],
                    test_output=script_df.loc[id]["output"],
                )
            else:
                list_of_fun_results, i = submit_functions(
                    functions_list=result_functions,
                    id=id,
                    ref=ref,
                    lang=lang,
                    outfolder=outfolder,
                    script_folder=script_df if script_df is not None else script_folder,
                    retry_mismatching_types=retry_mismatching_types,
                    roberta_mode=roberta_mode,  # if input program is untokenized, use True
                )

            ref_f_name = get_ref_f_name(script_df if script_df is not None else script_folder, id, result["lang"])

            status = [x[0] for x in list_of_fun_results]
            all_statuses.append(status)
            all_results.append(list_of_fun_results[0])

            if len(ref) > 0 and lang.lower() == "java":
                result_stats["static_total"] += 1
                ref = ref.replace("f_filled", ref_f_name)
                static_analysis_results, func_type_translation = static_analysis(
                    functions_list=result_functions, ref=ref, lang=lang,
                )
                for i in range(len(list_of_fun_results)):
                    if i < len(static_analysis_results):
                        list_of_fun_results[i] = [*list_of_fun_results[i], static_analysis_results[i]]
                    else:
                        list_of_fun_results[i] = [*list_of_fun_results[i], static_analysis_results[-1]]
                for src_type, tgt_types in func_type_translation.items():
                    type_translation[src_type].update(list(tgt_types))

            output_results[id] = list_of_fun_results
            nb_success = status.count("success")
            nb_idential = sum([x[0] == "success" and x[1] == "identical to gold" for x in list_of_fun_results])

            assert nb_success <= 1, "Should stop after first success"
            if nb_success > 0:
                result_stats["success"] += 1
                if nb_idential > 0:
                    result_stats["identical_gold"] += 1
            else:
                result_stats[status[0]] = result_stats.get(status[0], 0) + 1

            def get_success_number(list_result, i):
                try:
                    res = int(list_result[i][1].split("/")[0])
                except:
                    res = int(list_result[i][1][list_result[i][1].find("#Results:")+9:].split(",")[0])
                return res

            partial_status = [
                True if status[i] == "failure" and get_success_number(list_of_fun_results, i) else False
                for i in range(len(status))
            ]
            n_partial_success += bool(partial_status[0])
            n_partial_success_10 += bool(any(partial_status))
            if status[0] == "success":
                n_success += 1
                n_partial_success += 1
            if "success" in status:
                n_success_10 += 1
                n_partial_success_10 += 1

            result_stats["total_evaluated"] = result_stats["total"] - result_stats["script_not_found"]

            total = max(1e-6, result_stats["total_evaluated"])
            acc = round(n_success * 100.0 / float(total), 1)
            acc_10 = round(n_success_10 * 100.0 / float(total), 1)
            pbar.set_description(
                f"CA@1:{acc}, CA@5: {acc_10}, n_suc: {n_success}, n_succ_5: {n_success_10}, "
                f"np_suc: {n_partial_success}, np_succ_5: {n_partial_success_10}, total: {total}"
            )

    result_stats["total_evaluated"] = result_stats["total"] - result_stats["script_not_found"]
    result_stats["success_at_1"] = n_success
    if lang.lower() in ["java", "python"]:
        result_stats.update(analysis_by_method(output_results))
    print(result_stats)

    out_filename = pathlib.Path(outfolder).parent.joinpath(
        f"{pathlib.Path(outfolder).name}_aggregated_output{'_'+criteria if criteria else ''}.json"
    )
    with open(out_filename, "w") as f:
        json.dump(output_results, f)

    return result_stats, output_results


def evaluate_signature_only(
    generation_results: Dict,
    script_path: str,
    outfolder: str,

):
    out_filename = pathlib.Path(outfolder).parent.joinpath(
        f"{pathlib.Path(outfolder).name}_aggregated_output.json"
    )
    exec_results = json.load(open(out_filename))
    type_translation = defaultdict(Counter)
    if os.path.isfile(script_path) and script_path.endswith(".csv"):
        script_df = pd.read_csv(script_path, index_col="TARGET_CLASS")
    else:
        script_df = None

    with tqdm(generation_results.items()) as pbar:
        for id, result in tqdm(generation_results.items()):
            result_functions = result["function"] if isinstance(result["function"], list) else [result["function"]]
            ref = get_ref(script_df if script_df is not None else script_path, id, result["lang"])
            # if target unit test not found for translated program, skip
            if ref is None or len(ref)==0:
                continue
            # TODO this only works for java for now
            ref_f_name = get_ref_f_name(script_df if script_df is not None else script_path, id, result["lang"])
            ref = ref.replace("f_filled", ref_f_name)
            lang = result["lang"]
            static_analysis_results, func_type_translation = static_analysis(
                functions_list=result_functions, ref=ref, lang=lang,
            )
            for i, res in enumerate(exec_results[id]):
                res.append(static_analysis_results[i])
            for src_type, tgt_types in func_type_translation.items():
                type_translation[src_type].update(list(tgt_types))

    result_stats = analysis_by_method(exec_results)
    print(result_stats)
    print("========= type translation ========")
    print(type_translation)
    with open(out_filename, "w") as f:
        json.dump(exec_results, f)


def load_results_multiple_format(input_dir):
    output = {}
    for f in os.listdir(input_dir):
        if f.endswith(".results.json") or not f.endswith(".json"):
            continue
        data = json.load(open(f"{input_dir}/{f}"))
        if data["language"] == "py":
            completions = [data["prompt"]+c for c in data["completions"]]
        else:
            raise NotImplementedError
        output[data["name"]] = {
            "lang": "python",
            "function": completions
        }
    return output


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(f"Roberta mode={args.roberta_mode}")
    # submit_generations(
    #     generation_results=json.load(open(args.results_file)),
    #     outfolder=args.outfolder,
    #     script_folder=args.script_folder,
    #     roberta_mode=args.roberta_mode,
    #     retry_mismatching_types=args.retry_mismatching_types,
    #     refactor_before=False
    # )

    ##### testing if gold java would pass the submission
    # gen_results = {}
    # lang_processor = LangProcessor.processors["java"](root_folder=TREE_SITTER_ROOT)
    # df = pd.read_csv("../data/transcoder_all_train/valid_java-exp.csv")
    # # df["java_code"] = df["java_code"].apply(lambda x: lang_processor.detokenize_code(lang_processor.detokenize_code(x)))
    # for i, row in df.iterrows():
    #     gen_results[row["TARGET_CLASS"]] = {"lang":"java","function":[row["java_code"]]}


    # data_dir = "../../MultiPL-EX/dump/cpp-py/transcoder_eval-cpp-py-completion"
    data_dirs = [
        # "../../MultiPL-EX/dump/cpp-py/transcoder_fixed_eval-cpp-py-TStyped-MTexplain-completion",
        "../../MultiPL-EX/dump/cpp-py/transcoder_eval-cpp-py-MTexplain-completion",
        "../../MultiPL-EX/dump/cpp-py/transcoder_eval-cpp-py-TSkeep-MTexplain-completion",
        "../../MultiPL-EX/dump/cpp-py/transcoder_eval-cpp-py-TStyped-MTexplain-completion",
    ]
    for data_dir in data_dirs:
        gen_results = load_results_multiple_format(data_dir)
        submit_generations(
            generation_results=gen_results,
            outfolder=data_dir,
            script_folder="../../CodeGenMirror/data/transcoder_evaluation_gfg",
            roberta_mode=False,
            refactor_before=False
        )

    # gen_results = json.load(
    #     open(
    #         "../dump/transcoder_all_train_dev/java/cobol-java/translation_output_codex_nosig.json"
    #     )
    # )
    # evaluate_signature_only(
    #     generation_results=gen_results,
    #     script_path="../../CodeGenMirror/data/transcoder_all_train",
    #     outfolder="../dump/transcoder_all_train_dev/python/cobol-python/translation_validation_codex"
    # )

