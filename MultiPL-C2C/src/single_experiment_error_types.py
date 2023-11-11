import pdb
from typing import Optional

import numpy as np
import pandas as pd
from pathlib import Path
import json
import itertools
import argparse
import gzip

import tiktoken


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


ERROR_TYPES = ["type_error", "undeclared_error", "other_syntax_error", "assertion_error", "runtime_error", "unhelpful"]

def estimator(n: int, c: int, k: int) -> float:
    """
    isntead of pass @ k rate, just calculate fraction
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    # return c / min(n, k)


def open_json(fpath: Path, mode: str):
    return gzip.open(fpath, mode + "t") if fpath.suffix == ".gz" else open(fpath, mode)


def completion_is_nearly_all_comments(script:str, func_sig_end="{", function_start="function", end_of_func="}", comment_start="#"):
    lines = script.split("\n")
    total_lines = 0
    total_comments = 0
    in_function=False
    for i,l in enumerate(lines):
        if l.startswith(end_of_func) and in_function:
            break

        if in_function:
            total_lines += 1
            if l.strip().startswith(comment_start):
                total_comments += 1

        if l.strip().endswith(func_sig_end) or l.strip().startswith(function_start):
            in_function = True

    if total_lines == total_comments:
        # pdb.set_trace()
        return True
    if total_lines >= 4 and total_lines-total_comments <= 1:
        # pdb.set_trace()
        return True
    return False





def get_java_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK" and len(result["stderr"]) == 0:
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["// your code", "// write code", "// write your", "// todo"]]):
            error.append(5)
        elif "AssertionError" in result["stderr"]:
            error.append(3)
        elif "incompatible types" in result["stderr"]:
            error.append(0)
        elif 'Exception in thread "main"' in result["stderr"]:
            error.append(4)
        elif any([w in result["stderr"] for w in ["cannot find symbol", "undefined", "undeclared"]]):
            error.append(1)
        else:
            error.append(2)
    return np.array(error)


def get_py_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK" and len(result["stderr"]) == 0:
            error.append(-1)
        elif "undefined" in result["stderr"]:
            error.append(1)
        elif "AssertionError" in result["stderr"]:
            error.append(3)
        # elif "incompatible types" in result["stderr"]:
        #     error.append(0)
        # elif any([w in result["stderr"] for w in ["cannot find symbol", "undefined", "undeclared"]]):
        #     error.append(1)
        else:
            error.append(2)
    return np.array(error)


def get_sh_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK" and len(result["stderr"]) == 0:
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["# your code", "# write code", "# write your", "# todo"]]) or \
            completion_is_nearly_all_comments(result["program"]):
            error.append(5)
        elif ("usage:" in result["stderr"].lower() or "not defined" in result["stderr"].lower()) and result["status"] == "Exception":
            error.append(1)
        elif result["stderr"] == '' in result["stderr"] and result["status"] == "Exception":
            error.append(3)
        elif "runtime error" in result["stderr"].lower():
            error.append(4)
        else:
            error.append(2)
    return np.array(error)


def get_r_error_types(results):
    # TODO needs to be rewritten to accomadate new tests
    error = []
    for result in results:
        if result["status"] == "OK" and len(result["stderr"]) == 0:
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["# your code", "# write code", "# write your", "# todo"]]):
            error.append(5)
        elif any([w in result["stderr"] for w in ["is not TRUE", "is not FALSE"]]):
            error.append(3)
        elif "type" in result["stderr"]:
            error.append(0)
        elif any([w in result["stderr"] for w in ["could not find function"]]):
            error.append(1)
        elif 'Execution halted' in result["stderr"]:
            error.append(2)
        else:
            error.append(4)
    return np.array(error)


def get_jl_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK":
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["# your code", "# write code", "# write your", "# todo"]]) or \
            completion_is_nearly_all_comments(
                result["program"], func_sig_end="XXXXXX", function_start="function", end_of_func="end"):
            error.append(5)
        elif " 0 errored, 0 broken" in result["stderr"] or "Test Failed at" in result["stdout"]:
            error.append(3)
        elif " type " in result["stdout"]:
            error.append(0)
        elif any([w in result["stdout"] for w in ["not defined"]]):
            error.append(1)
        elif ' 0 passed, 0 failed,' in result["stderr"] or "ERROR:" in result["stderr"] or result["stderr"]=="":
            error.append(2)
        else:
            error.append(4)
    return np.array(error)


def get_lua_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK":
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["-- your code", "-- write code", "-- write your", "-- todo"]]):
            error.append(5)
        elif "expected: " in result["stdout"]:  # assertion error
            error.append(3)
        elif "expected, got" in result["stdout"]:
            error.append(0)
        elif any([w in result["stdout"] for w in ["attempt to call a nil value", "attempt to index a nil value"]]):
            error.append(1)
        # elif ' 0 passed, 0 failed,' in result["stderr"] or "ERROR:" in result["stderr"] or result["stderr"]=="":
        #     error.append(2)
        else:  # no run time errors defined, all syntax
            error.append(2)
    return np.array(error)


def get_cpp_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK" and len(result["stderr"]) == 0:
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["// your code", "// write code", "// write your", "// todo"]]):
            error.append(5)
        elif "Assertion failed" in result["stderr"]:
            error.append(3)
        elif "incompatible" in result["stderr"]:
            error.append(0)
        elif any([w in result["stderr"] for w in ["no template named", "undeclared identifier", "no matching function", "no member named"]]):
            error.append(1)
        elif result["status"] == "Timeout" or any([w in result["stderr"] for w in ["out_of_range"]]):
            error.append(4)
        else:
            error.append(2)
    return np.array(error)


def get_rkt_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK":
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["; your code", "; write code", "; write your", "; todo"]]):
            error.append(5)
        elif all([w in result["stderr"] for w in ["expected:"]]):
            error.append(3)
        # elif " type " in result["stdout"]:
        #     error.append(0)
        elif any([w in result["stderr"] for w in ["unbound identifier", "not an identifier"]]):
            error.append(1)
        elif "ERROR" in result["stderr"]:
            error.append(4)
        else:
            error.append(2)
    return np.array(error)


def get_cs_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK":
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["// your code", "// write code", "// write your", "// todo"]]):
            error.append(5)
        elif all([w in result["stderr"] for w in ["Debug.Assert"]]):
            error.append(3)
        elif any([w in result["stdout"] for w in ["cannot convert", "Cannot implicitly convert type", "cannot be converted"]]):
            error.append(0)
        elif any([w in result["stdout"] for w in ["could not be found", "does not contain a definition"]]):
            error.append(1)
        elif "Problem.Main" in result["stderr"] and "Syntax error" not in result["stderr"]:
            error.append(4)
        else:
            error.append(2)
    return np.array(error)


def get_d_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK":
            if result["exit_code"]!=0:
                x=0
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["// your code", "// write code", "// write your", "// todo"]]):
            error.append(5)
        elif any([w in result["stderr"] for w in ["passed unittests", "AssertError"]]):
            error.append(3)
        elif any([w in result["stderr"] for w in ["incompatible types", "argument types"]]):
            error.append(0)
        elif 'core.exception' in result["stderr"]:
            error.append(4)
        elif any([w in result["stderr"] for w in ["not defined", "undefined", "no property"]]):
            error.append(1)
        else:
            error.append(2)
    return np.array(error)


def get_swift_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK" and len(result["stderr"]) == 0:
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["// your code", "// write code", "// write your", "// todo"]]):
            error.append(5)
        elif any([w in result["stderr"] for w in ["Assertion failed"]]):
            error.append(3)
        elif any([w in result["stderr"] for w in ["cannot convert"]]):
            error.append(0)
        # elif 'core.exception' in result["stderr"]:
        #     error.append(4)
        elif any([w in result["stderr"] for w in ["cannot find", "has no member"]]):
            error.append(1)
        else:
            error.append(2)
    return np.array(error)


def get_scala_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK" and len(result["stderr"]) == 0:
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["// your code", "// write code", "// write your", "// todo"]]):
            error.append(5)
        elif any([w in result["stderr"] for w in ["AssertionError"]]):
            error.append(3)
        elif any([w in result["stderr"] for w in ["Type Mismatch Error", "needs type"]]):
            error.append(0)
        elif 'java.lang.' in result["stderr"]:
            error.append(4)
        elif any([w in result["stderr"] for w in ["Not Found Error"]]):
            error.append(1)
        else:
            error.append(2)
    return np.array(error)


def get_pl_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK" and len(result["stderr"]) == 0:
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["# your code", "# write code", "# write your", "# todo"]]):
            error.append(5)
        # elif any([w in result["stderr"] for w in ["Type Mismatch Error", "needs type"]]):
        #     error.append(0)
        elif any([w in result["stderr"] for w in ["Undefined subroutine", "Can't locate", "without a package"]]):
            error.append(1)
        elif result["stderr"] == "":
            error.append(3)
        elif "ok!" in result["stdout"]:
            error.append(4)
        else:
            error.append(2)
    return np.array(error)


def get_rs_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK" and len(result["stderr"]) == 0:
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["// your code", "// write code", "// write your", "// todo"]]):
            error.append(5)
        elif any([w in result["stderr"] for w in ["mismatched types"]]):
            error.append(0)
        elif any([w in result["stderr"] for w in ["cannot find", "no method", "not implemented"]]):
            error.append(1)
        elif "thread 'main' panicked at" in result["stderr"]:
            if "assertion failed" in result["stderr"]:
                error.append(3)
            else:
                error.append(4)
        else:
            error.append(2)
    return np.array(error)

def get_go_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK" and len(result["stderr"]) == 0:
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["// your code", "// write code", "// write your", "// todo"]]):
            error.append(5)
        elif any([w in result["stdout"] for w in [" expected "]]):
            error.append(3)
        # elif any([w in result["stderr"] for w in ["cannot convert"]]):
        #     error.append(0)
        # elif 'core.exception' in result["stderr"]:
        #     error.append(4)
        elif any([w in result["stderr"] for w in ["undefined"]]):
            error.append(1)
        else:
            error.append(2)
    return np.array(error)


def get_rb_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK":
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["# your code", "# write code", "# write your", "# todo"]]):
            error.append(5)
        elif any([w in result["stdout"] for w in ["expected but was"]]):
            error.append(3)
        # elif any([w in result["stderr"] for w in ["cannot convert"]]):
        #     error.append(0)
        elif any([w in result["stdout"] for w in ["undefined method", "uninitialized constant"]]):
            error.append(1)
        elif 'Finished' in result["stdout"]:
            error.append(4)
        else:
            error.append(2)
    return np.array(error)


def get_php_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK":
            error.append(-1)
        elif any([w in result["program"].lower() for w in ["# your code", "# write code", "# write your", "# todo", "// your code", "// write code", "// write your", "// todo"]]):
            error.append(5)
        elif any([w in result["stdout"] for w in ["Uncaught Exception: Test failed!"]]):
            error.append(3)
        # elif any([w in result["stderr"] for w in ["cannot convert"]]):
        #     error.append(0)
        elif any([w in result["stdout"] for w in ["Cannot redeclare", "undefined function"]]):
            error.append(1)
        # elif 'Finished' in result["stdout"]:
        #     error.append(4)
        else:
            error.append(2)
    return np.array(error)


def get_js_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK":
            error.append(-1)
        elif any([w in result["program"].lower() for w in [ "// your code", "// write code", "// write your", "// todo"]]):
            error.append(5)
        elif any([w in result["stderr"] for w in ["AssertionError"]]):
            error.append(3)
        # elif any([w in result["stdout"] for w in ["not assignable to"]]):
        #     error.append(0)
        # elif any([w in result["stdout"] for w in ["Cannot find"]]):
        #     error.append(1)
        # elif 'Finished' in result["stdout"]:
        #     error.append(4)
        else:
            error.append(2)
    return np.array(error)



def get_ts_error_types(results):
    error = []
    for result in results:
        if result["status"] == "OK":
            error.append(-1)
        elif any([w in result["program"].lower() for w in [ "// your code", "// write code", "// write your", "// todo"]]):
            error.append(5)
        elif any([w in result["stderr"] for w in ["AssertionError"]]):
            error.append(3)
        elif any([w in result["stdout"] for w in ["not assignable to"]]):
            error.append(0)
        # elif any([w in result["stdout"] for w in ["Cannot find"]]):
        #     error.append(1)
        # elif 'Finished' in result["stdout"]:
        #     error.append(4)
        else:
            error.append(2)
    return np.array(error)


def get_src_program_len(problem, type="char"):
    if isinstance(problem["translation_prompt"], str):
        prompt_str = problem["translation_prompt"]
    else:
        if "### python version" in problem["translation_prompt"][-1]["content"].lower():
            prompt_str = problem["translation_prompt"][-1]["content"]
        else:  # multi-turn prompt
            if len(problem["translation_prompt"]) % 4 == 0 and "summarize" in problem["translation_prompt"][1]["content"]:
                prompt_str = problem["translation_prompt"][-3]["content"]
            else:  # translation prompt
                prompt_str = problem["translation_prompt"][-1]["content"]

    src_lang = prompt_str.split("###")[1].split()[0]
    start = list(find_all(prompt_str, f"### {src_lang} version"))[-1]
    end = prompt_str.find("###", start + 1)
    code = "\n".join(prompt_str[start:end].split("\n")[2:])
    if type=="char":
        return len(code)
    elif type=="line":
        return len(code.split("\n"))
    elif type=="space_token":
        return len(code.split())
    elif type=="openai_token":
        enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
        return len(enc.encode(code))
    # elif type=="python_token":
    #     return len(code.split())
    else:
        raise NotImplementedError()


def understand_errors(data):
    n = len(data["results"])
    if "go" in data['language']:
        data["language"] = "go"
    errors = globals()[f"get_{data['language']}_error_types"](data['results'])
    # res = [[estimator(n, sum(errors[:k]==i), k) for i, err_name in enumerate(ERROR_TYPES)] for k in [1, 10, 100]]
    res = [[estimator(n, sum(errors == i), 1) for i, err_name in enumerate(ERROR_TYPES)]]
    return np.array(res)


def get_result_breakdown_by_len(result_array, all_data, buckets=4, len_type="char"):
    # result array is shape [#problem X 5]
    def acc(lst):
        return sum(lst) / (len(lst)+1e-20)

    lens = [get_src_program_len(d, len_type) for d in all_data]
    quartiles = pd.Series(lens).quantile(np.linspace(0,1,buckets+1)[1:-1]).tolist()
    quartiles = [0] + quartiles + [float('inf')]
    bin_idx = pd.cut(lens, bins=quartiles, labels=list(range(buckets)))
    res = [acc(1-result_array[bin_idx==q].sum(1)) for q in range(buckets)]
    return res


def get_all_data(d: str):
    all_data = []
    for p in itertools.chain(Path(d).glob("*.results.json"), Path(d).glob("*.results.json.gz")):
        with open_json(p, "r") as f:
            data = json.load(f)
            all_data.append(data)
    return all_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, help="Temperature completions were made at. \
                        If 0.2 runs pass@1 rather than pass@10 and pass@100", required=True)
    parser.add_argument("dirs", type=str,  help="Directories with results. ", nargs="+")
    args = parser.parse_args()
    k = 1 if args.temperature == 0.2 else 10

    for d in args.dirs:
        result_array = np.array([understand_errors(p) for p in get_all_data(d)]).squeeze()
        if len(result_array) < k:
            continue
        name = d.split("/")[-1] if d.split("/")[-1] != "" else d.split("/")[-2]

        print(f"Dataset,k,quartile_1,quartile_2,quartile_3,quartile_4")
        result_by_len = get_result_breakdown_by_len(result_array, get_all_data(d))
        print(f"{name},1," + ",".join([f"{r:.3f}" for r in result_by_len]))

        print(f"Dataset,k,{','.join(ERROR_TYPES)}")
        result = result_array.mean(axis=0)
        print(f"{name},1," + ",".join([f"{r:.3f}" for r in result]))


if __name__ == "__main__":
    main()
