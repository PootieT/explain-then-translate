import os
import argparse
import json
from pathlib import Path
from typing import *
import regex as re

from tqdm import tqdm
import numpy as np

TOFILL_TOKEN = "{{{TOFILL}}}"
MULTI_INTERMEDIATE_GENERATION_FLAG = "----MULTI_INTERMEDIATE_GENERATION_FLAG----"
FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}
SHORT2CANONICAL = {
    "java": "java",
    "py": "python",
    "sh": "bash",
    "r": "r",
    "jl": "julia",
    "lua": "lua",
    "cpp": "c++",
    "d": "d",
    "go": "go",
    "js": "javascript",
    "rs": "rust",
    "php": "php",
    "pl": "perl",
    "rb": "ruby",
    "rkt": "racket",
    "scala": "scala",
    "swift": "swift",
    "ts": "typescript",
    "cs": "c#"
}
CANONICAL2SHORT = {v: k for k, v in SHORT2CANONICAL.items()}
SHORT2CANONICAL["go_test.go"] = "go"

FEW_SHOT_EXAMPLES=["HumanEval_107_even_odd_palindrome", "HumanEval_126_is_sorted", "HumanEval_1_separate_paren_groups", "HumanEval_88_sort_array"]
WRONG_PROGRAM_DIR = Path(__file__).absolute().resolve().parents[1].joinpath("datasets", "humaneval_multi_wrong")
GOLD_PROGRAM_DIR = Path(__file__).absolute().resolve().parents[1].joinpath("datasets", "humaneval_multi")
ALL_PROGRAM_DIR = Path(__file__).absolute().resolve().parents[1].joinpath("datasets", "humaneval_multi_all")


def cap(s: str) -> str:
    if "script" in s:
        s = s.replace("script", "Script")
    if s == "php":
        return "PHP"
    return s[0].upper() + s[1:]

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_prompt_str(problem):
    return problem if isinstance(problem, str) else \
        problem["translation_prompt"] if "translation_prompt" in problem and isinstance(problem["translation_prompt"], str) \
        else problem["translation_prompt"][0] if "translation_prompt" in problem \
        else problem["prompt"] if isinstance(problem["prompt"], str) \
        else problem["prompt"][-1]["content"]


def get_source_code_from_prompt(problem, section_index=-2):
    prompt_str = get_prompt_str(problem)

    # only works with python as source
    # prompt_str.find(f"### {cap(SHORT2CANONICAL[problem['language']])} version")
    # return prompt_str[prompt_str.find("def"):
    #                   prompt_str.find(f"### {cap(SHORT2CANONICAL[problem['language']])} version")]
    section_indices = list(find_all(prompt_str, "### "))
    end = section_indices[section_index+1]
    start = prompt_str.find("\n", section_indices[section_index]) + 1
    return prompt_str[start: end].strip() + "\n\n"


def get_source_lang_from_prompt(problem):
    prompt_str = get_prompt_str(problem)
    return prompt_str[prompt_str.find("### "):].split()[1].strip()


def get_target_lang_from_prompt(problem):
    prompt_str = get_prompt_str(problem)
    start = list(find_all(prompt_str, "### "))[-1]
    return prompt_str[start:].split()[1].strip()

def get_python_signature_from_code(script):
    for l in script.split("\n"):
        if l.strip().startswith("def"):
            return l.strip()


def get_target_signature_from_prompt(problem):
    prompt_str = get_prompt_str(problem)
    start_str = f"### {cap(SHORT2CANONICAL[problem['language']])} version\n\n"
    start_list = list(find_all(prompt_str, start_str))
    if len(start_list) == 0:  # no signature, likely transcoder dataset
        return ""
    start = start_list[-1]
    tgt_sig = prompt_str[start + len(start_str):]
    return tgt_sig


def get_and_remove_java_docstring_from_prompt(tgt_sig_all):
    start = tgt_sig_all.find('{')
    assert start != -1, "docstring needs to be in source code for summary to be included as intermediate step"
    docstr = tgt_sig_all[start+1: tgt_sig_all.find("\n    public static")]
    src_code = tgt_sig_all.replace(docstr, "")
    docstr_lines = docstr.split("\n")
    while docstr_lines[-1].strip() == "" or docstr_lines[-1].strip().endswith(":"):
        docstr_lines = docstr_lines[:-1]
    docstr = " ".join([l.replace("//", "").strip() for l in docstr_lines])
    return docstr, src_code


def lang_specific_touchup(content: str, lang: str):
    if lang in ["C++", "Bash"]:
        content += "\n}"
    return content


def replace_token(code: str, token_map: Dict[str, str]):
    for src_token, tgt_token in token_map.items():
        code = str(re.sub(r"\b{}\b".format(src_token), tgt_token, code))
    return code

def extract_multi_views(multi_view_files, src_lang, tgt_lang, no_header=False, no_code=False):
    if not multi_view_files:
        return ""

    multi_views = {}
    src_lang, tgt_lang = src_lang + " version", tgt_lang + " version"
    not_include_sections = {src_lang, tgt_lang}
    if no_code:
        not_include_sections.update([cap(l)+" version" for l in CANONICAL2SHORT.keys()])
    for file in multi_view_files:
        if not os.path.isfile(file):
            return ""
        data = json.load(open(file))
        if "results" in file:
            data["completions"] = json.load(open(file.replace("results.","")))["completions"]
        content = data["translation_prompt"] + data["completions"][0]
        sections = {p[:p.find("\n")]: p[p.find("\n")+1:] for p in content.split("### ")[1:]}
        not_include_sections.update(multi_views.keys())
        sections_unique = {k: lang_specific_touchup(v, k.replace("version", "").strip()) for k, v in sections.items()
                           if k not in not_include_sections}
        multi_views.update(sections_unique)

    multi_view_str = ""
    for title, content in multi_views.items():
        if no_header:
            multi_view_str += f"{content.strip()}\n\n"
        else:
            multi_view_str += f"### {title}\n\n{content.strip()}\n\n"

    return multi_view_str.replace("This  ", "This ").strip()


def reorder_python_function_calls(func_calls: List[str]):
    """ Basically topological sort, put all sub-calls ahead of parent calls
    ex: ['sorted([s.swapcase() for s in s_list]', 's.swapcase()'] -> ['s.swapcase()', 'sorted([s.swapcase() for s in s_list]']
    Given the way we extract the function calls, they will be almost sorted, so
    going over the list once, greedily move violations around is fine.
    """
    for i in range(1, len(func_calls)):
        func_call = func_calls[i]
        # if current call is found as a substring of any previous component, move the current func call
        # to before the first call which contained the current call
        prev_contain = [func_call in prev_func_call for prev_func_call in func_calls[:i-1]]
        if any(prev_contain):
            first_index = prev_contain.index(True)
            func_calls = func_calls[:first_index] + [func_call] + [func_calls[first_index]] + func_calls[i+1:]
    return func_calls


def exclude_easy_python_function_call(func_call: str):
    func = get_method_name(func_call)
    return func in {"list", "print", "dict", "not", "append", "zip"}#, "range", "str", "int", "float", "filter"}


def fix_parenthesis(func_call: str):
    start = func_call.find("(")
    depth = 1
    i = 0
    for i, c in enumerate(func_call[start+1:]):
        if c == "(":
            depth += 1
        if c == ")":
            depth -= 1
        if depth == 0:
            break
    func_call = func_call[:start+i+2]

    # clean up brackets,
    idx = func_call.find("[")
    if idx > -1 and ("]" not in func_call) and (idx < func_call.find("(")):
        func_call = func_call[idx + 1:]
    return func_call


def extract_recursive_function_calls(func_call: str):
    """"""
    pattern = r"(([\w\d\[\]\.]*\.)*([\w][\w\d]*)\(.*\))"
    sub_calls = []
    not_expanded_calls = [func_call]
    while not_expanded_calls:
        curr_call = not_expanded_calls[0]
        curr_call_partial = curr_call[curr_call.find("(")+1:]
        matches = re.findall(pattern, curr_call_partial)
        if not matches:
            return sub_calls

        # if there are matches, fix parenthesis, append them to head of non-expanded calls
        # this way it ensures order in which they expand, leaf calls always ahead of parent calls
        added_calls = []
        for match in matches:
            added_calls.append(fix_parenthesis(match[0]))

        not_expanded_calls.remove(curr_call)
        not_expanded_calls = [*added_calls, *not_expanded_calls]
        sub_calls = [*added_calls, *sub_calls]
    return sub_calls


def get_method_name(func_call: str):
    return func_call.split("(")[0].split(".")[-1].strip()


def is_in_comment(code, start):
    i = start
    in_comment = False
    while code[i] != "#" and code[i] != "\n":
        if code[i - 1] == "#":
            in_comment = True
        i -= 1
    return in_comment


def extract_python_function_calls(code: str):
    """ A function to extract all function calls in python. This include built-in calls ('sorted()') and
    also calls that belong to an object (`list.reverse()`). We remove ones that are defined within the code
    and also ones that are very common (`list()`)"""
    # regexr.com/7d90l
    # pattern = r"[\w\d\[\]\.]*\.*[\w][\w\d]*[\s]*\(.*\)"
    pattern = r"(([\w\d\[\]\.]*\.)*([\w][\w\d]*)\(.*\))"   # enforce calls have ( right after method name. not syntactically necessary but most of model outputs obey this rule
    matches = re.findall(pattern, code)
    func_calls = []
    defined_funcs = set([])
    unique_methods = set([])
    for match in matches:
        match, _, method = match
        start = code.find(match)
        if code[start-4:start].startswith("def"):  # function definition, ignore
            defined_funcs.add(get_method_name(match))
            continue

        if is_in_comment(code, start):
            continue

        # collect all child calls, put them in the front
        func_call_candidates = extract_recursive_function_calls(match)
        real_match = fix_parenthesis(match)
        if real_match not in func_call_candidates:
            func_call_candidates.append(real_match)
        for func_call_candidate in func_call_candidates:
            if exclude_easy_python_function_call(func_call_candidate):
                continue

            # in addition to removing defined function call, we also remove duplicating methods. we assume same
            # method call translation can be applied to other instances, but this could be
            # flawed because the type of input argument could be different.
            method = get_method_name(func_call_candidate)
            if method in defined_funcs.union(unique_methods):
                continue

            func_calls.append(func_call_candidate)
            unique_methods.add(method)

    return func_calls


def rebalance_brackets(script: str, add_cnt: int, symbol="}"):
    if add_cnt >= 0:
        script += "\n" + symbol * add_cnt
    elif add_cnt < 0:
        while add_cnt < 0:
            if script[-1] == symbol:
                script = script[:-1].strip()
                add_cnt += 1
            else:
                script = script[:-1].strip()
    return script


def remove_pre_function_comments(script: str, is_func: Callable, is_comment):
    # remove preceeding comments to before class definition
    script_lines = script.split("\n")
    # main_func_name = self.get_function_name(prompt.strip().split("\n")[-1].strip())
    for i, line in enumerate(script_lines):
        if is_func(line):  # and self.get_function_name(line) == main_func_name:
            if all([is_comment(l) or l.strip() == "" or l.strip().startswith("###") for l in script_lines[:i]]):
                script_lines = script_lines[i:]
                script = "\n".join(script_lines)
                break
    return script


def remove_trailing_comments(script: str, end="}", is_comments=None):
    # after rebalancing brackets, remove everything that occurs after the last closing bracket
    # often this is a good enough heuristics to remove trailing chatGPT comments, or unit test
    # statements. The only failing case so far is if any comments come after contains "}" symbol
    if script.strip().endswith(end):
        return script

    # if is_comment is None, remove everything that comes after the end token
    if is_comments is None:
        i = -1
        while i > -len(script) + 10 and script[i-len(end):i] != end:
            i -= 1
        if script[i-len(end):i] == end:
            script = script[:i + 1] + "\n"
    else:  # otherwise, remove only commented lines after the last end token
        script_lines = script.split("\n")
        i = len(script_lines) - 1
        while i >= 0:
            if script_lines[i].endswith(end) or not (is_comments(script_lines[i]) or script_lines[i].strip()==""):
                break
            else:
                script_lines = script_lines[:i]
                i -= 1
        script = "\n".join(script_lines)
    return script


def fix_indentation(script, is_comment, indentation_level=1):
    # find the first non-comment statement, that would be the output base indentation level
    indent = "    "
    script_lines = script.split("\n")
    first_index = 0
    while first_index < len(script_lines) and is_comment(script_lines[first_index]) or len(script_lines[first_index].strip()) == 0:
        first_index += 1
    if script_lines[first_index].startswith(indent):
        completion_indentation_level = 1
    else:
        completion_indentation_level = 0

    need_to_indent = False
    if not script.startswith(indent):
        script = indent + script
        need_to_indent = True
    script_lines = script.split("\n")
    # add indent to following lines as long as previous lines have been comments
    top_comments = is_comment(script_lines[0]) and need_to_indent
    comment_idx = 1
    while top_comments and comment_idx < len(script_lines):
        top_comments = is_comment(script_lines[comment_idx]) and top_comments
        if top_comments:
            script_lines[comment_idx] = indent + script_lines[comment_idx]
        comment_idx += 1

    # and if base indentation level is 0, keep indenting the rest of the code
    # TODO technically we should dedent if desired indentation level is less, but no use for that for now
    if completion_indentation_level == 0 and indentation_level == 1:
        comment_idx -= 1
        while comment_idx < len(script_lines):
            script_lines[comment_idx] = "    " + script_lines[comment_idx]
            comment_idx += 1
    elif completion_indentation_level == 1 and indentation_level == 0:
        script_lines = [l[4:] if l.startswith(indent) else l for l in script_lines]
    script = "\n".join(script_lines)
    return script


def get_multi_vew_acronym(multi_view_dirs):
    str_list = []
    for d in multi_view_dirs:
        d = d.split("/")[-1]
        if "MT" in d:
            str_list.append("-".join(d[d.find("MT")+2:].split("-")[:2]).replace("-completion", ""))
        else:
            str_list.append(d[d.find("python")+7:].split("-")[0])
    return "_".join(sorted(str_list))


def get_gold_src_programs(src_lang: str, original_name:str, translator) -> Optional[Dict]:
    dataset_path = "datasets/humaneval_multi" if "originals" in original_name.lower() else "datasets/mbpp_multi"
    dataset_path = Path(__file__).parents[1].resolve().joinpath(dataset_path)
    data = json.load(open(dataset_path.joinpath(src_lang + ".json")))
    data = {name: translator.remove_tests(code) for name, code in data.items()}
    return data


def remove_tests(script: str, test_line: str, end_of_program: str):
    script_lines = script.split("\n")
    code_lines = []
    is_test = False
    for l in script_lines:
        if test_line in l:
            is_test = True
        if not is_test:
            code_lines.append(l)
    return "\n".join(code_lines+[end_of_program]).strip() + "\n"


def sample_random_wrong_program(
    translator,
    wrong_programs: Dict[str, Dict[str, int]],
    name: str,
    num_samples: int=1
):
    if name not in wrong_programs:
        return None
    programs, weights = [], []
    for p, c in wrong_programs[name].items():
        programs.append(p)
        weights.append(c)
    weights = [w/sum(weights) for w in weights]
    sampled_programs = np.random.choice(programs, num_samples, p=weights)
    sampled_programs = [translator.remove_tests(p) for p in sampled_programs]
    return sampled_programs


def get_wrong_programs(dataset, lang):
    assert "originals" in dataset, "Only accomadating humaneval dataset for now"
    data_path = f"{WRONG_PROGRAM_DIR}/{lang}.json"
    with open(data_path) as f:
        data = json.load(f)
    return data


def get_all_programs(dataset, lang):
    assert "originals" in dataset, "Only accomadating humaneval dataset for now"
    data_path = f"{ALL_PROGRAM_DIR}/{lang}.json"
    with open(data_path) as f:
        data = json.load(f)
    return data


def get_gold_program(translator, dataset, lang, name):
    assert "originals" in dataset, "Only accomadating humaneval dataset for now"
    data_path = f"{GOLD_PROGRAM_DIR}/{lang}.json"
    with open(data_path) as f:
        data = json.load(f)

    program = translator.remove_tests(data[name]) if name in data else None
    return program


def infer_name_from_gold(translator, dataset, lang, code):
    assert "originals" in dataset, "Only accomadating humaneval dataset for now"
    data_path = f"{GOLD_PROGRAM_DIR}/{lang}.json"
    with open(data_path) as f:
        data = json.load(f)

    code2name = {translator.remove_tests(v).strip(): k for k, v in data.items()}
    name = code2name.get(code)
    return name


def ablate_few_shot(translator, problems: List[Dict], shots_to_ablate: int, num_samples: int=1, is_retrieval=False, false_only=True):
    print(f"ablating few-shot problems: shots_to_ablate={shots_to_ablate}, num_samples={num_samples}, "
          f"is_retrieval={is_retrieval}, false_only={false_only}")
    if false_only:
        ablate_programs = get_wrong_programs(problems[0]["original"].split("/")[-2], problems[0]["language"])
    else:
        ablate_programs = get_all_programs(problems[0]["original"].split("/")[-2], problems[0]["language"])
    programs_to_del = []
    bar = tqdm(problems)
    for problem in bar:
        bar.set_description(f"Problem: {problem['name']}")
        sections = [[p[:p.find("\n")], p[p.find("\n") + 1:]] for p in problem["translation_prompt"].split("### ")[1:]]
        turn_len = 3 if "explain" in problem["translation_prompt"] else 2
        assert len(sections) % turn_len == 0
        num_shots = (len(sections) // turn_len) - 1
        ablate_indices = np.random.choice(num_shots, shots_to_ablate, replace=False)
        alt_prompt = [problem["translation_prompt"]]*num_samples
        # for each few shot examples we want to swap with a bad example
        for ablate_idx in ablate_indices:
            old_shot_target_idx = ablate_idx * turn_len + turn_len - 1
            old_shot_target = sections[old_shot_target_idx][1]
            # for each sample of bad program, we create a new prompt, so inference can do round-robin using each prompts
            # here we have to try to infer what problem it is, guessing name from program, since few-shot does not have
            # to be fixed
            few_shot_name = infer_name_from_gold(translator, problem["original"].split("/")[-2], problem["language"],
                                                 old_shot_target.strip()) if is_retrieval \
                            else FEW_SHOT_EXAMPLES[ablate_idx]
            if few_shot_name is None:
                raise IndexError("Cannot find few-shot in gold programs")
            alt_shot_targets = sample_random_wrong_program(translator, ablate_programs, few_shot_name,
                                                           num_samples)
            if alt_shot_targets is None:  # if no wrong programs found for this shot, remove problem from results
                programs_to_del.append(problem["name"])
                continue
            for sample_idx in range(num_samples):
                if false_only:
                    if old_shot_target.strip() not in alt_prompt[sample_idx]:
                        x = 1
                    elif old_shot_target.strip() == alt_shot_targets[sample_idx].strip():
                        x=1
                # assert alt_prompt[sample_idx] == alt_prompt[sample_idx].replace(old_shot_target.strip(), alt_shot_targets[sample_idx].strip())
                alt_prompt[sample_idx] = alt_prompt[sample_idx].replace(old_shot_target.strip(), alt_shot_targets[sample_idx].strip())
        problem["translation_prompt"] = alt_prompt[0] if num_samples == 1 else alt_prompt

    problems = [p for p in problems if not p["name"] in programs_to_del]
    return problems


if __name__ == "__main__":
    code = """#Python program to print topological sorting of a DAG
from collections import defaultdict
 
#Class to represent a graph
class Graph:
    def __init__(self,vertices):
        self.graph = defaultdict(list) #dictionary containing adjacency List
        self.V = vertices #No. of vertices
 
    # function to add an edge to graph
    def addEdge(self,u,v):
        self.graph[u].append(v)
 
    # neighbors generator given key
    def neighbor_gen(self,v):
        for k in self.graph[v]:
            yield k
     
    # non recursive topological sort
    def nonRecursiveTopologicalSortUtil(self, v, visited,stack):
         
        # working stack contains key and the corresponding current generator
        working_stack = [(v,self.neighbor_gen(v))]
         
        while working_stack:
            # get last element from stack
            v, gen = working_stack.pop()
            visited[v] = True
             
            # run through neighbor generator until it's empty
            for    next_neighbor in gen:
                if not visited[next_neighbor]:  # not seen before?
                    # remember current work
                    working_stack.append((v,gen))
                    # restart with new neighbor
                    working_stack.append((next_neighbor, self.neighbor_gen(next_neighbor)))
                    break
            else:
                # no already-visited neighbor (or no more of them)
                stack.append(v)
                 
    # The function to do Topological Sort.
    def nonRecursiveTopologicalSort(self):
        # Mark all the vertices as not visited
        visited = [False]*self.V
         
        # result stack
        stack = []
 
        # Call the helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if not(visited[i]):
                self.nonRecursiveTopologicalSortUtil(i, visited,stack)
        # Print contents of the stack in reverse
        stack.reverse()
        print(stack)
 
g= Graph(6)
g.addEdge(5, 2);
g.addEdge(5, 0);
g.addEdge(4, 0);
g.addEdge(4, 1);
g.addEdge(2, 3);
g.addEdge(3, 1);
 
print("The following is a Topological Sort of the given graph")
g.nonRecursiveTopologicalSort()"""
    code = """from typing import List
from typing import Tuple
from typing import Optional
def largest_smallest_integers(lst: List[int]) -> Tuple[Optional[int], Optional[int]]:
    smallest = list(filter(lambda x: x < 0, lst))
    largest = list(filter(lambda x: x > 0, lst))
    return (max(smallest) if smallest else None, min(largest) if largest else None)"""
    code = """from typing import List

def order_by_points(nums: List[int]) -> List[int]:
    def digits_sum(n):
        neg = 1
        if n < 0: n, neg = -1 * n, -1
        n = [int(i) for i in str(n)]
        n[0] = n[0] * neg
        return sum(n)
    return sorted(nums, key=digits_sum)"""
    code = """def circular_shift(x: int, shift: int) -> str:
    s = str(x)
    if shift > len(s):
        return s[::-1]
    else:
        return s[len(s) - shift:] + s[:len(s) - shift]"""
    code = """def max_fill(grid: List[List[int]], capacity: int) -> int:
    return sum([math.ceil(sum(arr)/capacity) for arr in grid])"""
    extract_python_function_calls(code)