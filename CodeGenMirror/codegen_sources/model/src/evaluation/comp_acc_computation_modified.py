# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import re
import subprocess
from pathlib import Path

import json
from concurrent.futures import ProcessPoolExecutor

from typing import List

# from evaluation.judge_utils import execute_code
# from ..dummy_executor import DummyExecutor
import pdb

import sys

import os

from ..utils import (
    REPO_ROOT,
    limit_virtual_memory,
    MAX_VIRTUAL_MEMORY,
    read_file_lines,
    get_java_bin_path,
)

sys.path.append(str(REPO_ROOT))
print("adding to path", str(REPO_ROOT))
TREE_SITTER_ROOT = REPO_ROOT.joinpath("tree-sitter")
import codegen_sources.preprocessing.lang_processors.cpp_processor
import codegen_sources.preprocessing.lang_processors.java_processor
import codegen_sources.preprocessing.lang_processors.python_processor
import codegen_sources.preprocessing.lang_processors.cobol_processor
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor

from codegen_sources.test_generation.test_runners.cpp_test_runner import CppTestRunner
from codegen_sources.test_generation.test_runners.python_test_runner import (
    PythonTestRunner,
)
from codegen_sources.test_generation.evosuite_tests_translators.evosuite_to_python import (
    EvosuiteToPython,
)
from codegen_sources.test_generation.evosuite_tests_translators.evosuite_to_cpp import (
    EvosuiteToCpp,
)

EXT = {"python": "py", "java": "java", "cpp": "cpp", "cobol": "cbl"}

TOFILL = {"python": "#TOFILL", "java": "//TOFILL", "cpp": "//TOFILL"}

primitive_types = {"short", "int", "long", "float", "double", "boolean", "char"}

EVOSUITE_TESTS_TRANSCODER_PATH = (
    REPO_ROOT.joinpath("data").joinpath("evosuite_unit_tests").joinpath("transcoder_test_set.json")
)

TIMEOUT_LIMIT = 15  # seconds (originally 120)

lang2ext={
    "python": "py",
    "java": "java",
    "cpp": "cpp",
    "cobol": "cbl",
    "fortran": "f90"
}
def find_all(a_str, sub):
    if len(sub) == 0:
        return []
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def eval_state(proc, proc_name, expected_output=None):
    try:
        try:
            result, stderr = proc.communicate(timeout=TIMEOUT_LIMIT)  # 120
        except subprocess.TimeoutExpired:
            c = "kill `ps aux | grep '" + proc_name + "' | grep -v jupyter | grep -v grep | awk '{print($2)}'`"
            subprocess.run(c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return "timeout", None
        results = result.decode("utf8", errors="replace")
        if "JUnit" in results:
            return eval_junit_state(results)
        if expected_output is not None:
            return eval_ibm_state(results, expected_output, stderr)
        if len(results) > 0:
            success, n_test = results.split("#Results:")[-1].split(",")
            if int(success) == int(n_test):
                return "success", None
            else:
                return "failure", result.decode("utf-8", errors="replace")
        else:
            # for special failed transcoder train python unit tests output are empty but stderr is good
            stderrs = stderr.decode("utf-8", errors="replace")
            if "SyntaxError" in stderrs:
                return "error", stderrs
            elif "OK" in stderrs:
                return "success", None
            else:
                test_output = re.search(
                    "Ran ([0-9]+) tests in [0-9.]+s\\n\\nFAILED \(errors=([0-9]+)",
                    stderrs,
                )
                if test_output:
                    n_test, fails = test_output.groups()
                    return (
                        "failure",
                        f"{int(n_test) - int(fails)} / {n_test} test cases passed",
                    )
                else:
                    return "error", stderrs
    except KeyboardInterrupt:
        raise
    except:
        return "error", stderr.decode("utf-8", errors="replace")


def eval_junit_state(results: str):
    success_cnt = re.search(r"(\d) tests successful", results).group(1)
    total_cnt = re.search(r"(\d) tests found", results).group(1)
    if success_cnt == total_cnt:
        return "success", None
    else:
        return "failure", f"{success_cnt}/{total_cnt} tests passed.",


def eval_ibm_state(results: str, expected_output: str, stderr: bytes):
    if results.strip() == expected_output.strip() or expected_output.strip() in results.strip():
        return "success", None
    else:
        stderr = stderr.decode("utf-8", errors="replace")
        if stderr and not results:
            return "error", stderr
        expected_outs = expected_output.split("\n")
        res_list = results.split("\n")
        success_cnt = 0
        total_cnt = len(expected_outs)
        for i, exp in enumerate(expected_outs):
            if len(res_list) > i and res_list[i] == exp:
                success_cnt += 1
        return "failure", f"{success_cnt}/{total_cnt} tests passed.",


def eval_cobol_state(proc, proc_name):
    try:
        try:
            result, stderr = proc.communicate(timeout=TIMEOUT_LIMIT)  # 120
        except subprocess.TimeoutExpired:
            c = "kill `ps aux | grep '" + proc_name + "' | grep -v jupyter | grep -v grep | awk '{print($2)}'`"
            subprocess.run(c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return "timeout", None
        results = result.decode("utf8", errors="replace")
        success_cnt = len(re.findall(r"success", results))
        failure_cnt = len(re.findall(r"failure", results))

        if len(result) > 0:
            if success_cnt > 0 and failure_cnt == 0:
                return "success", None
            else:
                return (
                    "failure",
                    f"{success_cnt}/{success_cnt+failure_cnt} tests passed.",
                )
        else:
            return "error", stderr.decode("utf-8", errors="replace")
    except KeyboardInterrupt:
        raise
    except:
        return "error", stderr.decode("utf-8", errors="replace")


def eval_cobol_state_judge(judge_res):
    # https://ce.judge0.com/#statuses-and-languages
    try:
        status_id = judge_res.get("status", {}).get("id", 6)
        if status_id == 5:
            return "timeout", None
        if status_id == 3:
            results = judge_res.get("stdout", "")
            success_cnt = len(re.findall(r"success", results))
            failure_cnt = len(re.findall(r"failure", results))

            if len(results) > 0:
                if success_cnt > 0 and failure_cnt == 0:
                    return "success", None
                else:
                    return (
                        "failure",
                        f"{success_cnt}/{success_cnt+failure_cnt} tests passed. {results}",
                    )
        else:
            return "error", (judge_res["compile_output"] or "") + (judge_res["stderr"] or "")
    except KeyboardInterrupt:
        raise
    except:
        return "error", (judge_res["compile_output"] or "") + (judge_res["stderr"] or "")


def eval_fortran_transpile_state(proc, proc_name):
    try:
        try:
            result, stderr = proc.communicate(timeout=TIMEOUT_LIMIT)  # 120
        except subprocess.TimeoutExpired:
            c = "kill `ps aux | grep '" + proc_name + "' | grep -v jupyter | grep -v grep | awk '{print($2)}'`"
            subprocess.run(c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return "timeout", None
        results = result.decode("utf8", errors="replace")
        if "filed with exit status" in results:
            return "error", stderr.decode("utf-8", errors="replace")
        else:
            return "success", ""
    except KeyboardInterrupt:
        raise
    except:
        return "error", stderr.decode("utf-8", errors="replace")


def run_python_program(script_path, i):
    proc = subprocess.Popen(
        f"{limit_virtual_memory(MAX_VIRTUAL_MEMORY)}; python {script_path}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        executable="/bin/bash",
    )
    res = eval_state(proc, f"python {script_path}")
    return res, i


def run_java_program(script_path, i):
    if "EvoRunner" in open(script_path).read():
        return run_java_evosuite_program(script_path, i)
    folder = os.path.dirname(script_path)
    name = os.path.basename(script_path).split(".")[0]
    proc = subprocess.Popen(
        f"{limit_virtual_memory(MAX_VIRTUAL_MEMORY)}; cd {folder} &&  {os.path.join(get_java_bin_path(), 'javac')} {name}.java && {os.path.join(get_java_bin_path(), 'java')} {name}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        executable="/bin/bash",
    )
    res = eval_state(proc, f"java {name}")
    return res, i


def run_java_evosuite_program(script_path, i):
    folder = os.path.dirname(script_path)
    name = os.path.basename(script_path).split(".")[0]
    extra_env = os.environ
    extra_env["JUNIT_HOME"] = f"{extra_env['HOME']}/java"
    extra_env["PATH"] = f"{extra_env['PATH']}:{extra_env['JUNIT_HOME']}"
    extra_env["CLASSPATH"] = f"{extra_env.get('CLASSPATH', '.:.*')}:{extra_env['JUNIT_HOME']}/junit-4.13.2.jar:{extra_env['JUNIT_HOME']}/hamcrest-core-1.3.jar"
    proc = subprocess.Popen(
        f"{limit_virtual_memory(MAX_VIRTUAL_MEMORY)}; cd {folder} &&  {os.path.join(get_java_bin_path(), 'javac')} {name}.java && {os.path.join(get_java_bin_path(), 'java')} -jar {Path().absolute().parent.joinpath('junit-platform-console-standalone-1.9.1.jar')} --class-path . -c {name.replace('.java', '')}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        executable="/bin/bash",
        env=extra_env
    )
    res = eval_state(proc, f"java {name}")
    return res, i


def run_java_ibm_program(script_path, i, test_input, test_output):
    folder = os.path.dirname(script_path)
    name = os.path.basename(script_path).split(".")[0]
    if not isinstance(test_input, str):
        stdin=False
    else:
        stdin=True
        open("tmp.txt", "w").write(test_input)  # TODO not parallel safe
    proc = subprocess.Popen(
        f"{limit_virtual_memory(MAX_VIRTUAL_MEMORY)}; cd {folder} &&  {os.path.join(get_java_bin_path(), 'javac')} {name}.java && {os.path.join(get_java_bin_path(), 'java')} {name}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=open("tmp.txt","r") if stdin else None,
        shell=True,
        executable="/bin/bash",
    )
    res = eval_state(proc, f"java {name} && {test_input}", expected_output=test_output)
    return res, i


def run_cpp_program(script_path, i):
    folder = os.path.dirname(script_path)
    name = os.path.basename(script_path).split(".")[0]
    proc = subprocess.Popen(
        f"{limit_virtual_memory(MAX_VIRTUAL_MEMORY)}; cd {folder} && g++ {name}.cpp -o {name}_cpp && ./{name}_cpp",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        executable="/bin/bash",
    )
    res = eval_state(proc, f"{name}_cpp")
    return res, i


def run_cobol_program(script_path, i):
    folder = os.path.dirname(script_path)
    name = os.path.basename(script_path).split(".")[0]
    proc = subprocess.Popen(
        f"{limit_virtual_memory(MAX_VIRTUAL_MEMORY)}; cd {folder} && cobc -x --free {name}_test.cbl {name}.cbl && ./{name}_test",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        executable="/bin/bash",
    )
    res = eval_cobol_state(proc, f"cobc {name}")
    return res, i


def run_cobol_program_judge(script_path, i):
    exec_res = execute_code(
        open(script_path).read(),
        lang_id=77,
        cpu_time_limit=TIMEOUT_LIMIT,
        wall_time_limit=20.0,  # max judge allowed time
        memory_limt=MAX_VIRTUAL_MEMORY / 1024,  # unit in kb
    )
    res = eval_cobol_state_judge(exec_res)
    return res, i


def run_fortran_program(script_path, i):
    # compile fortran to python language first
    folder = os.path.dirname(script_path)
    file_name = os.path.basename(script_path)
    program_path = file_name.replace(f"_test.py", ".f90")
    program_out = program_path.replace(".f90", "")
    proc = subprocess.Popen(
        f"{limit_virtual_memory(MAX_VIRTUAL_MEMORY)}; cd {folder} && python -m numpy.f2py -c {program_path} -m {program_out}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        executable="/bin/bash",
    )
    res = eval_fortran_transpile_state(proc, f"python -m numpy.f2py -c {program_path} -m {program_out}")
    if res[0] == "error":
        return res, i
    proc = subprocess.Popen(
        f"{limit_virtual_memory(MAX_VIRTUAL_MEMORY)}; python {script_path}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        executable="/bin/bash",
    )
    res = eval_state(proc, f"python {script_path}")
    return res, i


def modify_evosuite_tests(script_model, f, id, f_name):
    # f = f.replace("f_filled", f_name)  # instead of renaming functions back to original name, we need to rename tests
    f = f"class {id[:-10]} {{\n\t{f}}}"
    script = re.sub(f"{id}\.([\w]+)\(", f"{id}.f_filled(", script_model)
    script = script.replace(TOFILL["java"], f)

    # actual program's class will be id[:-10], and test class will be full id
    script = script.replace(f"{id} extends", f"temp_test extends")
    script = script.replace(id, id[:-10])
    script = script.replace(f"temp_test extends", f"{id} extends")



    script = f"import junit.framework.TestCase;\nimport org.junit.Test;\nimport static org.junit.Assert.*;\nimport java.util. *;\nimport java.util.stream.*;\nimport java.lang.*;\n\n" + script
    return script


def make_arg_string(argtype, argval):
    if "[" not in argtype:
        return f"{argtype} {argval}"

    dim = argtype.count("[")
    argtype = argtype.replace("[", "").replace("]", "")
    return f'{argtype} {argval} {"[ ]" * dim}'


def convert_filled_arguments(script_model, f, lang, lang_processor, f_name=None):
    """
    script_model is expected to be the full evaluation script including headers, comments
    f_gold functions, and test cases.
    """

    assert lang in {"java", "cpp"}
    header = []
    arguments_gold = lang_processor.extract_arguments(script_model)
    # return_type_gold = get_return_type(script_model)
    return_type_gold = get_gold_return_type(script_model)

    arguments_filled = lang_processor.extract_arguments(f)
    return_type_filled = get_return_type(f)

    if arguments_gold[0] == arguments_filled[0]:
        return None
    if f_name is None:
        f_name = lang_processor.get_function_name(f)

    argument_types_gold = [t.strip() for t in arguments_gold[0]]
    arguments_strings = [make_arg_string(arg_type, f"param{i}") for i, arg_type in enumerate(argument_types_gold)]
    new_function_lines = [
        f'static {return_type_gold} f_filled({", ".join(arguments_strings)})',
        "{",
    ]

    new_params_strings = []
    for param_index, (param_type_gold, param_type_filled) in enumerate(zip(argument_types_gold, arguments_filled[0])):
        param_type_filled = param_type_filled.strip()
        param_type_gold = param_type_gold.strip()
        if param_type_filled == param_type_gold:
            new_params_strings.append(f"param{param_index}")
        elif lang == "cpp":
            if "vector" in param_type_filled:
                if "int" not in argument_types_gold:
                    return None
                ints_indices = [i for i, t in enumerate(argument_types_gold) if t == "int" and i > param_index]
                if any([i > param_index for i in ints_indices]):
                    array_length_arg = min([i for i in ints_indices if i > param_index])
                else:
                    array_length_arg = min(ints_indices)
                new_function_lines.append(
                    f'{param_type_filled.replace("&", "")} vect_param{param_index}(param{param_index}, param{param_index} + param{array_length_arg});'
                )
                new_params_strings.append(f"vect_param{param_index}")
            elif param_type_filled == "string" and "char" in param_type_gold:
                new_function_lines.append(
                    f'{param_type_filled.replace("&", "")} string_param{param_index}(param{param_index});'
                )
                new_params_strings.append(f"string_param{param_index}")
            elif param_type_gold == "string" and "char" in param_type_filled:
                new_function_lines.append(f"char char_arr_param{param_index}[param{param_index}.length() + 1];")
                new_function_lines.append(f"strcopy(char_arr_param{param_index}, param{param_index}.c_str());")
                new_params_strings.append(f"char_arr_param{param_index}")
            else:
                new_params_strings.append(f"({param_type_filled}) param{param_index}")
        elif lang == "java":
            if (
                param_type_filled == "String" and "char" in param_type_gold
            ) or param_type_filled == transform_to_java_object_type(param_type_gold):
                new_params_strings.append(f"{param_type_filled}.valueOf(param{param_index})")
                # header.append("#include <cstring>")  # may need to move to cpp block?
            elif param_type_gold == "String":
                new_params_strings.append(f"param{param_index}.toCharArray()")
            else:
                new_params_strings.append(f"({param_type_filled}) param{param_index}")
        else:
            return None

    inner_function_name = "f_filled_inner"
    outer_f_return_string = f'{inner_function_name}({",".join(new_params_strings)})'
    if return_type_filled != return_type_gold:
        outer_f_return_string = f"({return_type_gold}) {outer_f_return_string}"
    new_function_lines += [f"return {outer_f_return_string};", "}"]

    f = lang_processor.detokenize_code(f.replace(f_name, inner_function_name))
    return "\n".join(list(set(header))) + script_model.replace(TOFILL[lang], "\n".join([f, "\n"] + new_function_lines))


def submit_evosuite_functions(functions_list, id, lang, test_dictionary, roberta_mode=False):
    # pdb.set_trace()
    assert lang in {"cpp", "python"}, f"{lang} is not supported for evosuite tests"
    if lang == "cpp":
        test_runner = CppTestRunner(timeout=30, compilation_timeout=30)
    else:
        assert lang == "python"
        test_runner = PythonTestRunner(timeout=30)
    lang_processor = LangProcessor.processors[lang](root_folder=TREE_SITTER_ROOT)
    id = id.rstrip()
    if id not in test_dictionary or test_dictionary[id] == "missing":
        return [return_script_not_found()], id
    test = test_dictionary[id]
    results_list = []
    for try_id, f_fill in enumerate(functions_list):
        f = f_fill.rstrip()
        f = lang_processor.detokenize_code(f) if not roberta_mode else f.replace("#NEWLINE", "\n")
        result = test_runner.get_tests_results(f, test)
        results_list.append((result[0], None))
        if result[0] == "success":
            return results_list, id
    return results_list, id


def submit_functions(
    functions_list,
    id,
    ref,
    lang,
    outfolder,
    script_folder,
    retry_mismatching_types,
    roberta_mode=False,
):
    # pdb.set_trace()
    results_list = []
    i = id.rstrip()

    lang_processor = LangProcessor.processors[lang](root_folder=TREE_SITTER_ROOT)
    new_function_list = []
    for f in functions_list:
        try:
            new_function_list.append(" ".join(lang_processor.tokenize_code(f)))
        except Exception as e:
            new_function_list.append(f)
    functions_list = new_function_list

    for try_id, f_fill in enumerate(functions_list):
        f = f_fill.rstrip()
        script_model_path = os.path.join(script_folder, f"{lang}/{i}.{EXT[lang]}")
        if os.path.exists(script_model_path):
            script_model = open(script_model_path, "r", encoding="utf-8").read()
            # if "f_gold" in script_model:
            #     script_model = script_model[script_model.find("TOFILL")-2:]
            try:
                f_name = lang_processor.get_function_name(f)
                # carefully replace f_name, and not to replace function call of same name
                # function from a different module: skip calls proceeded with "."
                idx = f.find(f_name)
                while idx != -1:
                    if "." not in f[idx-2:idx]:
                        f = f[:idx] + "f_filled" + f[idx+len(f_name):]
                    idx = f.find(f_name, idx+1)
                # f = f.replace(f_name, "f_filled")
            except:
                results_list.append(("error", "Could not replace function name"))
                f_name = "default_backup_function_name"
            f = lang_processor.detokenize_code(f).strip() if not roberta_mode else f.replace("#NEWLINE", "\n").strip()
            # make sure tokenization isn't the only issue
            if f == ref or f == lang_processor.detokenize_code(lang_processor.tokenize_code(ref)) or f.replace("f_filled", f_name) == ref:
                results_list.append(("success", "identical to gold"))
                return results_list, i

            if "EvoRunner" in script_model and lang == "java":   # evosuite test in java
                script = modify_evosuite_tests(script_model, f, id, f_name)
            else:
                script = script_model.replace(TOFILL[lang], f)
            if lang == "python":
                script = f"import numpy as np \nimport math\nfrom math import *\nimport collections\nfrom collections import *\nfrom typing import *\nimport heapq\nimport itertools\nimport random\nimport sys\nimport bisect\nimport datetime\nimport unittest\n\n{script}"
            script_path = f"{outfolder}/{i}.{EXT[lang]}"
            open(script_path, "w", encoding="utf-8").write(script)
            run_pg = globals()[f"run_{lang}_program"]
            result, _ = run_pg(script_path, i)
            if result[0] == "success":
                results_list.append(result)
                return results_list, i
            elif retry_mismatching_types and lang in {"cpp", "java"}:
                try:
                    script_transform_args = convert_filled_arguments(
                        script_model, f_fill, lang, lang_processor, f_name=f_name
                    )
                except KeyboardInterrupt:
                    raise
                except:
                    script_transform_args = None

                if script_transform_args is not None:
                    open(script_path, "w", encoding="utf-8").write(script_transform_args)
                    run_pg = globals()[f"run_{lang}_program"]
                    result2, _ = run_pg(script_path, i)
                    if result2[0] == "success":
                        results_list.append(result2)
                        return results_list, i
                    else:
                        result = (
                            result2[0],
                            "".join(
                                [
                                    result[1] if result[1] else "",
                                    f"|| second run handling types mismatch: ## function ## {script_transform_args} ## output ## {result2[1]}",
                                ]
                            ),
                        )

            results_list.append(result)
        else:
            return [return_script_not_found()], i
    return results_list, i


def submit_cobol_functions(
    functions_list: List[str],
    id: str,
    outfolder: str,
    script_folder: str,
    run_all_beams: bool = False,
    refactor_before: bool = False,
):
    results_list = []
    i = id.rstrip()

    for try_id, f_fill in enumerate(functions_list):
        f = f_fill.rstrip()
        script_model_path = os.path.join(script_folder, f"cobol/{i}.cbl")
        if os.path.exists(script_model_path):
            script_model = open(script_model_path, "r", encoding="utf-8").read()
            try:
                f_name = re.search(r"PROGRAM-ID[ ]?. ([^\.]+)[ ]?\.", f).group(1)
                f = f.replace(f_name, "f_filled")
            except:
                results_list.append(("error", "Could not replace function name"))
            if refactor_before:
                f = refactor_cobol_program(f)

            # local execution (old)
            test_path = f"{outfolder}/{i[:25]}_test.cbl"
            program_path = f"{outfolder}/{i[:25]}.cbl"
            open(test_path, "w", encoding="utf-8").write(script_model)
            open(program_path, "w", encoding="utf-8").write(f)
            result, _ = run_cobol_program(program_path, i)

            # judge0 api submission (new)
            # program_path = f"{outfolder}/{i[:25]}.cbl"
            # open(program_path, "w", encoding="utf-8").write(f"{script_model}\n\n{f}")
            # result, _ = run_cobol_program_judge(program_path, i)

            if result[0] == "success":
                results_list.append(result)
                if not run_all_beams:
                    return results_list, i

            results_list.append(result)
        else:
            return [return_script_not_found()], i
    return results_list, i


def get_java_class_name(code):
    start = code.find(" class ") + 7
    end = code.find("{", start)
    return code[start:end].strip()

def submit_ibm_functions(
    functions_list: List[str],
    id: str,
    lang: str,
    outfolder: str,
    test_input: str,
    test_output: str,
    run_all_beams: bool = False,
):
    results_list = []
    i = id.rstrip()

    for try_id, f_fill in enumerate(functions_list):
        f = f_fill.rstrip()

        if lang == "java":
            method_name = f.split("(")[0].split()[-1].strip()
            f = f.replace(method_name, "main")
            script = f"import java.util.Scanner;\nimport java.util.*;\nimport java.util.stream.*;\nimport java.lang.*;\n\npublic class {i} {{\n{f}\n}}"
            script_path = f"{outfolder}/{i}.java"
        else:
            script=f
            script_path = f"{outfolder}/{i}.{lang2ext[lang]}"
        open(script_path, "w", encoding="utf-8").write(script)
        result, _ = globals()[f"run_{lang}_ibm_program"](script_path, i, test_input, test_output)

        if result[0] == "success":
            results_list.append(result)
            if not run_all_beams:
                return results_list, i

        results_list.append(result)
    return results_list, i


def submit_fortran_functions(
    functions_list: List[str],
    id: str,
    outfolder: str,
    script_folder: str,
    run_all_beams: bool = False,
):
    results_list = []
    i = id.rstrip()
    processor = LangProcessor.processors["fortran"](root_folder=TREE_SITTER_ROOT)
    for try_id, f_fill in enumerate(functions_list):
        f = f_fill.rstrip()
        script_model_path = os.path.join(script_folder, f"python/{i}.py")
        if os.path.exists(script_model_path):
            script_model = open(script_model_path, "r", encoding="utf-8").read()
            script_model = f"import numpy as np \nimport math\nfrom math import *\nimport collections\nfrom collections import *\nfrom typing import *\nimport heapq\nimport itertools\nimport random\nimport sys\nimport bisect\nimport datetime\nimport unittest\nfrom {i} import f_filled\n\n{script_model}"
            try:
                f_name = processor.get_function_name(f)
                f = f.replace(f_name, "f_filled")
            except:
                results_list.append(("error", "Could not replace function name"))

            # local execution (old)
            test_path = f"{outfolder}/{i}_test.py"
            program_path = f"{outfolder}/{i}.f90"
            open(test_path, "w", encoding="utf-8").write(script_model)
            open(program_path, "w", encoding="utf-8").write(f)
            result, _ = run_fortran_program(test_path, i)

            if result[0] == "success":
                results_list.append(result)
                if not run_all_beams:
                    return results_list, i

            results_list.append(result)
        else:
            return [return_script_not_found()], i
    return results_list, i

def refactor_cobol_program(code: str) -> str:
    codelines = code.strip().split("\n")
    while len(codelines) > 1 and codelines[-1].strip().startswith(("#", "//", "*")) or len(codelines[-1].strip()) == 0:
        codelines = codelines[:-1]

    # not necessary if subprogram is called from different file, but if two programs are combined in single file
    # the lack of this line will cause compile error
    if "end program f_filled." not in codelines[-1].lower():
        codelines.append("END PROGRAM f_filled.")

    codelines = [reformat_cobol_comment_styles(l) for l in codelines]
    code = "\n".join(codelines) + "\n"

    # stop run stops the main program from sub-program, no no
    code = code.replace("STOP RUN", "GOBACK").replace("stop run", "GOBACK")

    code = code.replace("&gt;", ">").replace("&lt;", "<").replace("&le;", "<=").replace("&ge;", ">=")

    return code


def reformat_cobol_comment_styles(codeline: str) -> str:
    line = codeline.strip()
    # gnuCobol compiler specific syntax for comment : *> instead of *
    if line and line[0] == "*":
        codeline = codeline.replace("*", "*>", 1)
    elif line.startswith("//"):
        codeline = codeline.replace("//", "*>", 1)
    elif line.startswith("#"):
        codeline = codeline.replace("#", "*>", 1)
    elif line.startswith("</") and line.endswith(">"):  # </code>
        codeline = ""
    return codeline


def eval_function_output(
    ref_path,
    hyp_paths,
    id_path,
    lang2,
    outfolder,
    script_folder,
    retry_mismatching_types,
    roberta_mode,
    evosuite_functions=False,
    evosuite_tests=None,
):
    # pdb.set_trace()
    functions = list(zip(*[read_file_lines(path) for path in hyp_paths]))
    ids = read_file_lines(id_path)
    refs = read_file_lines(ref_path)
    assert len(functions) == len(ids), f"{len(functions), len(ids)}"
    assert len(functions) == len(refs), f"{len(functions), len(refs)}"
    lang = lang2.split("_")[0]
    jobs = []
    executor = ProcessPoolExecutor(240)  # DummyExecutor() #
    # todo, we can do a mini-refactor here (checkout a new branch)
    # and we can just return how long it took to execute and then do some more analysis
    for f, i, r in zip(functions, ids, refs):
        if evosuite_functions:
            jobs.append(
                executor.submit(
                    submit_evosuite_functions,
                    f,
                    i,
                    lang,
                    evosuite_tests[lang],
                    roberta_mode,
                )
            )
        else:
            jobs.append(
                executor.submit(
                    submit_functions,
                    f,
                    i,
                    r,
                    lang,
                    outfolder,
                    script_folder,
                    retry_mismatching_types,
                    roberta_mode,
                )
            )

    results_stats = {
        "success": 0,
        "failure": 0,
        "error": 0,
        "timeout": 0,
        "script_not_found": 0,
        "identical_gold": 0,
    }
    results = ["" for _ in range(len(ids))]
    for job in jobs:
        results_list, i = job.result()
        nb_success = sum([r[0] == "success" for r in results_list])
        nb_identical = sum([r[0] == "success" and r[1] == "identical to gold" for r in results_list])
        assert nb_success <= 1, "Should stop after first success"
        if nb_success > 0:
            results_stats["success"] += 1
            if nb_identical > 0:
                results_stats["identical_gold"] += 1
        else:
            results_stats[results_list[0][0]] = results_stats.get(results_list[0][0], 0) + 1
        results[ids.index(i + "\n")] = []
        for result, stderr in results_list:
            if stderr is not None:
                stderr = stderr.replace("\n", " ")
            else:
                stderr = "None"
            results[ids.index(i + "\n")].append(f"{result} : {stderr}")

    results_stats["total"] = len(functions)
    results_stats["total_evaluated"] = len(functions) - results_stats["script_not_found"]
    results_stats = {k: results_stats[k] for k in sorted(results_stats.keys())}

    return results_stats, results


def load_evosuite_transcoder_tests(params=None):
    # cpp_test_translator = EvosuiteToCpp()
    python_test_translator = EvosuiteToPython()
    tests = {"java": {}, "java_scaffolding": {}, "python": {}, "cpp": {}}
    tests_path = params.translated_unit_tests_path if params is not None else EVOSUITE_TESTS_TRANSCODER_PATH
    with open(tests_path, "r") as f:
        for l in f:
            json_line = json.loads(l)
            if json_line["tests_strings"] == "missing":
                continue
            tests["java"][json_line["TARGET_CLASS"]] = json_line["tests_strings"]
            # tests["java_scaffolding"][json_line["TARGET_CLASS"]] = json_line[
            #     "scaffoldings_strings"
            # ]
            python_test = json_line.get("python_translated_tests")
            if python_test is None:
                python_test = python_test_translator.translate(json_line["tests_strings"])
            if not python_test_filter(python_test):
                continue
            tests["python"][json_line["TARGET_CLASS"]] = python_test

            # cpp_test = cpp_test_translator.translate(json_line["tests_strings"])
            # tests["cpp"][json_line["TARGET_CLASS"]] = cpp_test
    return tests


def python_test_filter(python_test):
    return python_test.count("try ") == 0 and python_test.count("catch(") == 0 and python_test.count("assert ") > 0


def return_script_not_found():
    return "script_not_found", None


def transform_to_java_object_type(t):
    if t not in primitive_types:
        return t
    if t == "int":
        return "Integer"
    if t == "char":
        return "Character"
    return t.capitalize()


def get_return_type(tokenized_java):
    return tokenized_java.split("(")[0].split()[-2]


def get_gold_return_type(tokenized_java_file: str) -> str:
    # compared to get_return_type, this expects input to be full script,
    # not just the java function
    idx = tokenized_java_file.find("f_gold")
    return_type = tokenized_java_file[idx - 10 : idx].split()[-1]
    return return_type
