import pdb
from typing import *
from pathlib import Path

import sys

from dataset_builder.utils import CANONICAL2SHORT

sys.path.append(str(Path(__file__).parents[1].joinpath("dataset_builder")))


def cap(s: str):
    return s[0].upper() + s[1:]


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches


def extract_code_from_encasing(completion: str) -> str:
    """ extract code from being enclosed in ``` code ``` (markdown)
    heuristics used is we extract all segments of codes, then pick the longest fragment
    (smaller fragments can be step by step translation)
    otherwise, if ``` does not exist, use heuristics
    """
    if "```" not in completion:
        sep_count = list(find_all(completion, "### "))
        # if there are multiple responses segments, pick first,
        if len(sep_count)>2:
            completion = completion[completion.find("\n", sep_count[0])+1: sep_count[1]]
        # if there is 2 sections, remove the steps section
        elif len(sep_count) == 2:
            # if steps is in the first segment title, remove first segment
            first_segment_end = completion.find("\n", sep_count[0]) + 1
            second_segment_end = completion.find("\n", sep_count[1]) + 1
            if "step" in completion[sep_count[0]: first_segment_end] and not "version" in completion[sep_count[0]: first_segment_end]:
                completion = completion[second_segment_end:]
            else:
                completion = completion[first_segment_end: sep_count[1]]
        # if there is one or zero, remove anything after ### if code comes first
        else:
            keys = ["here", "sure", "note", "certainly"]
            lines = [l for l in completion.split("\n") if not any(l.lower().strip().startswith(k) for k in keys)]
            for i, l in enumerate(lines):
                if "steps" in l.lower() and (":" in l or "###" in l) and i > 5:
                    lines = lines[:i]
                    break
            completion = "\n".join(lines)
        return completion
    # completion.replace("\n```", "").replace("```\n", "")
    codes = []
    code = []
    max_idx = 0
    for line in completion.split("\n"):
        if line.startswith("```"):
            # start of a new block of code,
            if len(code) == 0:
                code.append(" ".join(line.split()[1:]))
            else:
                # end of a block of code, compute length
                codes.append("\n".join(code))
                if len(codes[-1]) > len(codes[max_idx]):
                    max_idx = len(codes) - 1
                code = []
        else:
            # in the middle of code block
            if len(code) > 0:
                code.append(line)

    # if a code block is incomplete, add it as another block and calculate max index
    if len(code) > 0:
        codes.append("\n".join(code))
        if len(codes[-1]) > len(codes[max_idx]):
            max_idx = len(codes) - 1

    return codes[max_idx]


def remove_first_n_lines(completion:str, n=1) -> str:
    return "\n".join(completion.split("\n")[n:])


def cleanup_completion(completion: str, prompt: str) -> str:
    """ chat GPT response isn't always completion, sometimes it returns full-blown code
    Here we remove everything that is inserted before the prompt's last line (signature)
    Sometimes, it changes the arguments and types, so we need to be careful when removing

    assumptions: regardless if it's chat completion or regular completion
        - the main function to be filled is always

    """
    pdb.set_trace()
    # TODO this may not extend to langs other than java
    _, lang = get_langs_from_prompt(prompt)
    translator = __import__(f"humaneval_to_{CANONICAL2SHORT[lang.lower()]}").Translator()
    code_signature = prompt.strip().split("\n")[-1].strip()
    func_name = translator.get_function_name(code_signature)

    # extract just the code from NL response
    completion = extract_code_from_encasing(completion)

    # make sure functions are complete at the end, no artifacts from early stops
    completion = translator.remove_imports(completion)
    completion = translator.remove_class_decl(completion)
    completion = translator.completion_touchup(completion, prompt)
    functions = translator.extract_functions(completion)
    pdb.set_trace()
    if len(functions) == 0:
        # # if no functions found, it's a regular completion. In some langs we allow full functions to complete (java)
        # # in others, we don't (bash), hence we need to check if this is the function to be completed
        # if completion.startswith(code_signature):
        #     completion = remove_first_n_lines(completion)
        # else:  # in this case, signature is not found and rest of code is complete
        pass
    elif len(functions) == 1:
        # pdb.set_trace()
        # if one function found, it's either one function, or main function completion +
        # one helper function, or one function with one child-level function
        if not completion.startswith(functions[0]):
            # if top function is incomplete: leave it alone
            pass
        else:
            if func_name in functions[0].split("\n")[0]:
                # if top is an complete function, remove signature line
                # TODO record signature and check type/variable name inconsistency
                completion = remove_first_n_lines(completion)
            else:
                # one incomplete parent function with child-level function at top
                pass
    else:
        # pdb.set_trace()
        # if more than one function, it's either all complete functions, or completion + functions
        if not any([completion.startswith(f) for f in functions]):
            # if top function is incomplete: leave it alone
            pass
        else:
            main_func_idx = [i for i, func in enumerate(functions) if func_name in func.split("\n")[0]]
            main_func_idx = len(functions) - 1 if len(main_func_idx) == 0 else main_func_idx[0]
            if main_func_idx != 0:
                completion = "\n".join([functions[main_func_idx]] +
                                       [func for i, func in enumerate(functions) if i != main_func_idx])
            # TODO record signature and check type/variable name inconsistency
            completion = remove_first_n_lines(completion)

    return completion


def get_langs_from_prompt(prompt:str):
    if "Translate the following" in prompt:
        src_lang = prompt.replace("Translate the following", "").split()[0]
        tgt_lang = prompt[:prompt.find("given")].split()[-1]
    else:
        candidate_indices = list(find_all(prompt, "### "))
        src_lang = prompt[candidate_indices[0]:].split()[1:3][0]
        tgt_lang = prompt[candidate_indices[-1]:].split()[1:3][0]
        # src_lang, tgt_lang = None, None
        # for i, idx in enumerate(candidate_indices):
        #     split_words = prompt[idx:].split()[1:3]
        #     if split_words[1] == "version":
        #         if src_lang is None:
        #             src_lang = split_words[0]
        #         elif tgt_lang is None:
        #             tgt_lang = split_words[0]
        #             break

        # if only one instance found, it's target lang
        # if tgt_lang is None and src_lang is not None:
        #     src_lang, tgt_lang = "python", src_lang

        # src_lang = "python" if src_lang is None or src_lang == "commented" else src_lang
        # tgt_lang = "python" if tgt_lang is None or tgt_lang == "commented" else tgt_lang
    return cap(src_lang.strip()), cap(tgt_lang.strip())


def cleanup_completion_simple(completion: str, prompt: str) -> str:
    """
    Simple function to deal with outputs from completions to allow LMs to generate multiple functions
    but remove all function calls / natural language garbage / comments after the code
    very simple logic:
    - scan line by line
        - if found end of function:
            - if next none empty line declares a functioon,
                - continue
            otherwise remove all rest of the lines, and break
    We will initiate translator objects from each lang, if translator object implements both
        - is_function_signature (e.x. bash)
        - is_end_of_function
    or
        - additional_stops (e.x. racket)
    """
    # pdb.set_trace()
    if any(w in prompt for w in ["### Potential Explanations", "### Score"]):
        # this is a rerank completion, not actually generating program
        return completion.strip()

    _, lang = get_langs_from_prompt(prompt)
    translator = __import__(f"humaneval_to_{CANONICAL2SHORT[lang.lower()]}").Translator()
    completion = completion.replace("<|im_end|>", "").replace("<|im_sep|>", "")

    # for lispy like language where end of line could be at any place, we either have to count brackets
    # (which requires lexer, which sometimes isn't perfect), or we find a way to determine if a new line
    # is definitely NOT a function.
    # pdb.set_trace()
    if hasattr(translator, "additional_stops") and hasattr(translator, "get_function_name"):
        code_signature = prompt.strip().split("\n")[-1].strip()
        func_name = translator.get_function_name(code_signature)
        stops = translator.additional_stops.copy()
        stops.append(f"\n{func_name}")
        completion = truncate_after_additional_stops(completion, stops)

    # for languages that have distinct end of function token following conventional line based structure
    # we can detect by the last line's token
    if hasattr(translator, "is_function_signature") and hasattr(translator, "is_end_of_function"):
        completion = truncate_after_function_ends(completion, translator)
    # pdb.set_trace()
    return completion


def truncate_after_function_ends(completion, translator):
    lines = completion.split("\n")
    in_function = True
    for i, l in enumerate(lines):
        if in_function:
            # if encounter end of function, start looking out for next non empty line
            if translator.is_end_of_function(l):
                in_function = False
        else:  # if we are not in function, check if line is signature
            if len(l.strip()) != 0:
                if translator.is_function_signature(l):
                    in_function = True
                else:
                    lines = lines[:i]
                    break
    truncated_completion = "\n".join(lines)
    return truncated_completion


def truncate_after_additional_stops(completion, stops: List[str]):
    for stop in stops:
        if stop.lower() in completion.lower():
            completion = completion[:completion.lower().find(stop.lower())]
    return completion
