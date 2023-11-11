# This is a helper script for translating problems from the OpenAI HumanEval
# problems to Language L.
import ast
import pdb
import traceback
from glob import glob
import re
import os
import csv
import json
from pathlib import Path
import argparse

import pandas as pd

from base_language_translator import LanguageTranslator
from typing import List, Optional, Union, Dict

from codegen_sources.model.src.utils import TREE_SITTER_ROOT
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor

from dataset_builder.utils import SHORT2CANONICAL, cap, find_all, get_source_code_from_prompt, \
    get_target_signature_from_prompt, TOFILL_TOKEN, get_python_signature_from_code, extract_multi_views, \
    get_and_remove_java_docstring_from_prompt, extract_python_function_calls, replace_token, get_wrong_programs, \
    sample_random_wrong_program, get_gold_program


class FileWrapper:

    def __init__(self, name: str, program: str):
        self.name = name
        self.program = program

def translate_expr(translator, py_expr: ast.AST):
    """
    Translates a Python expression to Language L.
    """

    match py_expr:
        case ast.Constant(value=s):
            return translator.gen_literal(s)
        case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=n)) if type(3) in [int, float]:
            return translator.gen_literal(-n)
        case ast.Name(id):
            return translator.gen_var(id)
        case ast.List(elts=elts):
            return translator.gen_list([translate_expr(translator, e) for e in elts])
        case ast.Tuple(elts=elts):
            return translator.gen_tuple([translate_expr(translator, e) for e in elts])
        case ast.Dict(keys=keys, values=values):
            return translator.gen_dict(
                [translate_expr(translator, e) for e in keys],
                [translate_expr(translator, e) for e in values],
            )
        case ast.Call(func, args):
            return translator.gen_call(
                translate_expr(translator, func),
                [translate_expr(translator, a) for a in args],
            )
        case _other:
            # print("OMFG" + py_expr)
            raise Exception(f"Unhandled expression: {py_expr}")


class PromptVisitor(ast.NodeVisitor):
    """Helper for translate_prompt"""

    def __init__(self, translator):
        super().__init__()
        self.state = "start"
        self.translator = translator

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.state != "start":
            self.state = "error"
            return

        self.name = node.name
        self.args = node.args.args
        self.returns = node.returns

        match node.body:
            case [ast.Expr(value=ast.Constant(s)), ast.Pass()] if type(s) == str:
                self.description = s
                self.state = "complete"
            case _other:
                self.state = "error"
        
    def translate_func_decl(self, doctest_transformation: str, remove_description:bool=False) -> str | None:
        if self.state != "complete":
            return None
        match doctest_transformation:
            case "keep":
                desc = self.description
            case "remove":
                doctestRegex = re.compile(r'>>>.*\n.*\n')
                desc = re.sub(doctestRegex, '', self.description)
                if desc == self.description:
                    print('skipping (no doctests to remove)')
                    return None
            case "transform":
                # We first run the translate_prompt with the original
                # prompt. This is a hack! We need each script to have some sort
                # of setup method that remembers type information and such. But,
                # people have already done so in translate_prompt because it used
                # to be called first all the time. Calling it here as a setup
                # function should hopefully(!) not break anything
                self.translator.translate_prompt(self.name, self.args, self.returns, self.description)

                # Steps:
                # Find the Python expression and result in each doctest
                # py_ast = ast.parse("PYTHON EXPRESSION", "bogus filename")
                # translate_expr(py_ast, self.translator) to get the string for that expression in the target language
                
                #Split up the prompt from the doctests
                #promptAndDoctests = self.description.split('>>>')
                if '>>>' in self.description: #checking if there are doctests
                    doctestRegex = re.compile(r'>>>.*\n.*\n')
                    onlyDocTests = []
                    for m in re.finditer(doctestRegex, self.description):
                        onlyDocTests.append((m.start(),m.end()))
                    desc = ''
                    pos = 0
                    for i in onlyDocTests:
                        desc += self.description[pos:i[0]]
                        doctest = self.description[i[0]:i[1]]
                        doclist = doctest.split('\n') #Splitting up the output from the function call of the doctest
                        funcCall = ast.parse(doclist[0].strip('>>> ')).body[0].value
                        output = ast.parse(doclist[1].strip()).body[0].value
                        transl_funccall = translate_expr(self.translator, funcCall)
                        transl_output = translate_expr(self.translator, output)
                        if hasattr(self.translator, "finalize"):
                            transl_funccall = self.translator.finalize(transl_funccall, "lhs")
                            transl_output = self.translator.finalize(transl_output, "rhs")
                        # Why is this str() here?
                        desc += '>>> ' + transl_funccall + '\n    ' + str(transl_output) + '\n'
                        pos = i[1]
                    
                    desc += self.description[pos:]

                    # for test in (promptAndDoctests[1:]): #Removing each doctest from any junk
                    #     onlyDocTests.append(doctestRegex.match(test).group())
                    
                    # funcCalls = []
                    # outputs = []
                    # for doctest in onlyDocTests:
                    #     doclist = doctest.split('\n') #Splitting up the output from the function call of the doctest
                    #     funcCalls.append(ast.parse(doclist[0].strip()).body[0].value)
                    #     outputs.append(ast.parse(doclist[1].strip()).body[0].value)

                    # for i in range(len(funcCalls)):
                    #     funcCalls[i] = translate_expr(self.translator, funcCalls[i])
                    #     outputs[i] = translate_expr(self.translator, outputs[i])
                    
                    # desc = promptAndDoctests[0]
                    # for i in range(len(funcCalls)):
                    #     desc += funcCalls[i] + '\n' + outputs[i] + '\n\n'
                else: #else when there are no doctests
                    # Still return the description, because we are probably rewording!
                    desc = self.description
            case _other:
                raise Exception(f"bad doctest_transformation")
        if remove_description:
            desc = ""

        return self.translator.translate_prompt(self.name, self.args, self.returns, desc)


def translate_prompt(translator, doctest_transformation: str, py_prompt: str, filename: str, prompt_terminology: Optional[str]="keep") -> str:
    """
    Reads in a prompt from the HumanEval dataset with "    pass" appended. Translates the prompt to
    Language L. Ignores type annotations and imports. Fails if the prompt has auxiliary functions.
    """
    prompt_ast = ast.parse(py_prompt + "    pass", filename)
    prompt_visitor = PromptVisitor(translator)
    try:
        prompt_visitor.visit(prompt_ast)
        return prompt_visitor.translate_func_decl(doctest_transformation, prompt_terminology=="remove")
    except Exception as e:
        print(f"Exception translating prompt for {filename}: {e}")
        traceback.print_exception(e)
        return None


def translate_tests(translator, py_tests: str, entry_point: str, filename: str) -> str:
    """
    Translates a suite of tests from the HumanEval dataset to Language L. Expects the code to look like:

    METADATA = ... <-- optional

    def check():
        assert(LHS == RHS)
        ...
    """
    tests_ast = ast.parse(py_tests, filename)
    test_cases = translator.test_suite_prefix_lines(entry_point)
    match tests_ast:
        case ast.Module(body=[ast.FunctionDef(body=body)]):
            body_ast = body
        case ast.Module(body=[ast.Assign(), ast.FunctionDef(body=body)]):
            body_ast = body
        case _other:
            return None  # TODO: Should this blow up?
    for item_ast in body_ast:
        match item_ast:
            case ast.Assert(
                test=ast.Compare(left=left, ops=[ast.Eq()], comparators=[right])
            ):
                try:
                    left = translate_expr(translator, left)
                    right = translate_expr(translator, right)
                    if hasattr(translator, "finalize"):
                        left = translator.finalize(left, "lhs")
                        right = translator.finalize(right, "rhs")
                    test_cases.append(translator.deep_equality(left, right))
                except Exception as e:
                    print(f"Exception translating expressions for {filename}: {e}")
                    traceback.print_exception(e)
                    return None
            case ast.Expr(value=ast.Name(id="print")):
                pass
            case _other:
                print("Failed to translate tests for " + filename)
                return None
    for line in translator.test_suite_suffix_lines():
        test_cases.append(line)
    return "\n".join(test_cases)


def target_path(args, translator, file):
    file = Path(file).resolve()
    cleaned_task_id = re.search("HumanEval_\d+", file.name).group(0)
    entry_point = re.search("(HumanEval_\d+)_(.+).py", file.name).group(2)
    file_ext = get_file_ext_from_translator(translator)
    filename = Path(
        file.parent,
        "..",
        f"{file_ext}-{args.doctests}-{args.model}",
        f"{cleaned_task_id}_{entry_point}.{file_ext}",
    ).resolve()
    return filename


lang_dict = {}
with open('terms.csv','r') as of:
    term_list = csv.DictReader(of)
    for row in term_list:
        lang_dict[row['py']] = row
    fields = [k.strip() for k in row.keys()]

def consonant(s):
    return s.lower() not in 'aeiou'

def vowel(s):
    return s.lower() in 'aeiou'

def translate_terms(language,fields,prompt):
    """
    Takes a programming language name, a list of vocabulary words to translate, and a portion of docstring text.
    Returns the docstring text with Python-specific vocab translated to the target language.
    """
    if language == "go_test.go":
        language = "go"
    target_dict = lang_dict[language]
    for f in fields:
        if f in prompt and target_dict[f] != 'Q':
            if 'an '+f in prompt and consonant(target_dict[f][0]):
                prompt = prompt.replace('an '+f,'a '+target_dict[f])
            elif 'a '+f in prompt and vowel(target_dict[f][0]):
                prompt = prompt.replace('a '+f,'an '+target_dict[f])
            prompt = prompt.replace(f,target_dict[f])    #can't be an else: need to catch 2nd occurences of term that don't have article
    return prompt


def edit_prompt_terminology(language, example):
    """
    Takes a programming language name and the text of a python file.
    Translates Python-specific terms in natural language portions of the docstring to the target language.
    Returns the full text of the python file with translated natural language docstring.
    """
    before,prompt,after = example.replace("'''",'"""').split('"""')
    doctestRegex = re.compile(r'>>>.*\n.*\n')
    doctests = []
    for m in re.finditer(doctestRegex,prompt):
        doctests.append((m.start(),m.end()))
    if len(doctests) == 0:
        tar_prompt = translate_terms(language,fields,prompt)
    else:
        tar_prompt = ''
        last = 0
        for i in doctests:
            more_prompt = translate_terms(language,fields,prompt[last:i[0]])
            more_doctest = prompt[i[0]:i[1]]
            last = i[1]
            tar_prompt += more_prompt+more_doctest
        tar_prompt += translate_terms(language,fields,prompt[last:])

    return before+'"""'+tar_prompt+'"""'+after


def translate_prompt_and_tests(
    original_file,
    translator,
    doctests,
    prompt_terminology,
    source_program,
    target_signature,
    source_prompt,
    few_shot,
    shots,
    prompt_type="completion",
    src_lang="py",
    src_program=None,
    obfuscate=False
):
    entry_point = re.search("([^0-9]+_\d+)_(.+).py", original_file.name).group(2)
    reading_prompt = True
    reading_tests = False
    reading_source_program = False
    prompt_buffer = []
    tests_buffer = []
    source_program_buffer = []
    with open(original_file) as f:
        for line in f:
            if "### Canonical solution below ###" in line:
                reading_prompt = False
                reading_source_program = True
            if "### Unit tests below ###" in line:
                reading_tests = True
                reading_source_program = False
                continue
            if "def test_check():" in line:
                break

            if reading_prompt:
                prompt_buffer.append(line)
            if reading_tests:
                tests_buffer.append(line)
            if reading_source_program:
                source_program_buffer.append(line)

    tgt_lang = translator.__module__.split("_")[-1]
    src_program = src_program if src_lang != "py" else extract_python_source_code(
        doctests, prompt_buffer, prompt_terminology, source_program_buffer, source_prompt)
    if src_program is None:  # when there isn't a gold program on the source side (that isn't python)
        return None

    if obfuscate:
        assert src_lang == "py"  # , "C++", "Java", just so we can extract src_signature and redo
        transcoder_src_lang = "python" if src_lang == "py" else src_lang.lower()
        lang_processor = LangProcessor.processors[transcoder_src_lang](root_folder=TREE_SITTER_ROOT)
        obf_src_program, token_map = lang_processor.obfuscate_code(src_program)
        src_program = obf_src_program.strip() + "\n\n"
        token_map = {s.split()[1]: s.split()[0] for s in token_map.split(" | ")}
        prompt_buffer = [replace_token(l, token_map) for l in prompt_buffer]
        tests_buffer = [replace_token(l, token_map) for l in tests_buffer]
        if tgt_lang in ["php", "rkt"]:
            entry_point = "FUNC_0"  # test translator uses entry point to call translated function in tests

    prompt = "".join(prompt_buffer)
    if prompt_terminology == "reworded":
        prompt = edit_prompt_terminology(translator.file_ext(), prompt)
    translated_prompt = translate_prompt(translator, doctests, prompt, original_file.name, prompt_terminology)
    # When doctests == "remove" and there are no doctests in prompt, we get None.
    # If not, we could create a translated prompt that is identical to the
    # doctests == "keep" case.
    if translated_prompt is None:
        return None

    tests = "".join(tests_buffer)
    translated_tests = translate_tests(
        translator, tests, entry_point, original_file.name
    )

    if translated_tests is None:
        return None

    if target_signature != "keep":
        translated_prompt = ""

    if source_program != "remove":
        translated_prompt = add_source_code(translated_prompt, src_program , SHORT2CANONICAL[src_lang], tgt_lang)
        translated_prompt = add_few_shot(translated_prompt, few_shot, shots, tgt_lang, prompt_type)
        if translated_prompt is None:  # when current problem is one of few shot examples
            return None

    return translated_prompt, translated_tests


def remove_prompt_instruction(prompt: str):
    return prompt[:prompt.find('"""')].strip()


def add_source_code(prompt: str, src_code, source_lang, target_lang):
    """ add source code to context, also change the prompt to translation task """
    target_lang = SHORT2CANONICAL[target_lang] if target_lang in SHORT2CANONICAL else target_lang
    prompt_str = f"### {cap(source_lang)} version\n\n{src_code.strip()}\n\n" \
                 f"### {cap(target_lang)} version"
    if len(prompt) > 0:
        prompt_str += f"\n\n{prompt.strip()}\n"
    return prompt_str


def extract_python_source_code(doctests, prompt_buffer, prompt_terminology, source_program_buffer, source_prompt):
    stop = 0
    if source_prompt and prompt_terminology != "remove":
        if doctests == "keep":
            stop = len(prompt_buffer) - 1
        else:
            while not prompt_buffer[stop].startswith("    >>> "):
                stop += 1
            stop -= 1
    else:
        while not prompt_buffer[stop].startswith("def"):
            stop += 1
    source_lines = prompt_buffer[:stop + 1]
    if source_prompt and prompt_terminology != "remove" and doctests != "keep":
        source_lines += ['    """\n']
    source_lines += source_program_buffer[1:]
    source_code = "".join(source_lines).strip()
    return source_code


def add_few_shot(prompt_str, few_shots, shots, tgt_lang, prompt_type="completion"):
    lang = cap(SHORT2CANONICAL[tgt_lang])
    if len(few_shots) == 0:
        return prompt_str
    if isinstance(few_shots, list):
        assert few_shots[0]["role"] == "system"
        if shots == 0:
            few_shots = few_shots[:2]
            instruction = few_shots[1]["content"][:few_shots[1]["content"].find("###")]
            few_shots[1]["content"] = instruction + prompt_str
        else:
            assert 2*shots + 1 <= len(few_shots)
            few_shots = few_shots[:1+2*shots]
            assert few_shots[-1]["role"] == "assistant"
            few_shots.append({
                "role": "user",
                "content": prompt_str
            })
    elif isinstance(few_shots, pd.Series):  # pandas, retrieval prompts
        if prompt_type == "completion":
            # assert lang.lower() == "java"
            few_shot = f"You are a helpful and faithful compiler that transpiles Python code to {lang} code. Please translate the following Python code to {lang}?\n\n"
            retrieval_shot_idx, cur_idx = 1, 0
            while cur_idx < shots:
                if 'top'+str(retrieval_shot_idx)+'_src' not in few_shots:
                    raise IndexError("Not enough few-shot examples to cover nan gold target program, increase retrieval topk!")
                cur_shot_src = few_shots['top'+str(retrieval_shot_idx)+'_src']
                cur_shot_tgt = few_shots['top'+str(retrieval_shot_idx)+'_tgt']
                if isinstance(cur_shot_src, float) or isinstance(cur_shot_tgt, float):
                    retrieval_shot_idx += 1
                    continue
                few_shot += "### Python version\n\n" \
                            f"{cur_shot_src.strip()}\n\n" \
                            f"### {lang} version\n\n" \
                            f"{cur_shot_tgt.strip()}\n\n"
                retrieval_shot_idx += 1
                cur_idx += 1

            few_shot += prompt_str
            few_shots = few_shot
        else:
            instruction = f"Translate the following Python code to {lang} given the first parts of the {lang} program? Complete the rest from where the {lang} program left off. Keep the {lang} signature exactly the same as the code provided."
            few_shot = [
                {"role": "system", "content": f"You are a helpful and faithful compiler that transpiles Python code to {lang} code."}
            ]
            for i in range(shots):
                few_shot.extend(prepare_shots(few_shots[f"top{i}_src"], few_shots[f"top{i}_src"]))
            few_shot.append({
                    "role": "user",
                    "content": prompt_str
                })
            few_shot[1]["content"] = instruction + few_shot[1]["content"]
    else:  # string, regular completion prompt
        sep_indices = list(find_all(few_shots, "### ")) + [-1]
        few_shots_str = few_shots[:sep_indices[shots*2]].strip()
        if any([w in prompt_str for w in [
            "is_sorted",
            "even_odd_palindrome",
            "separate_paren_groups",
            "sort_array"
        ]]):
            x = 1
        if prompt_str.replace("\n\n\n","\n\n").replace(" \n","\n").strip() in few_shots_str.replace("\n\n\n","\n\n").replace(" \n","\n").strip():
            return None
        few_shots = few_shots_str + "\n\n" + prompt_str

    return few_shots


def prepare_shots(src_code, tgt_code):
    src_code = ""
    shots = [
        {"role": "user", "content": ""},
        {"role": "user", "assistant": ""},
    ]
    return shots

def get_stop_from_translator(translator) -> List[str]:
    if isinstance(translator, LanguageTranslator):
        return translator.stop()
    else:
        return translator.stop

def list_originals(root):
    key_func = lambda s: int(str(s.name).split("_")[1])
    directory = Path(Path(__file__).parent, "..", "datasets").resolve()
    files_unsorted = directory.glob(f"{root}/*.py")
    # assumption: base filenames are in the format of HumanEval_X_*.py
    # Where X is a valid number
    files_by_number = {key_func(file): file for file in files_unsorted}
    return files_by_number


def modify_translation_prompt(
    problem,
    multiturn_prompt,
    few_shots: Union[List, str, pd.Series],
    shots: int,
    src_lang="Python",
    prompt_type="completion",
    multi_view_files: Optional[List[str]]=None,
    multiturn_template_count: int=1,
) -> Optional[Dict]:
    if isinstance(problem["prompt"], str) and not ("translate" in problem["prompt"] or "### " in problem["prompt"]):
        return problem
    if multi_view_files and not all([os.path.isfile(f) for f in multi_view_files]):
        return None

    tgt_sig_all = get_target_signature_from_prompt(problem)
    src_code = get_source_code_from_prompt(problem)
    tgt_sig_line = tgt_sig_all.strip().split("\n")[-1]
    tgt_lang = cap(SHORT2CANONICAL[problem['language']])

    if multiturn_prompt == "single":
        problem["translation_prompt"] = problem["prompt"]
        problem["prompt"] = tgt_sig_all
        return problem

    if prompt_type == "completion":
        if multiturn_prompt == "steps":
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you summarize this {src_lang} program into a few steps such that a reader can easily rewrite the program in other languages?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### steps\n\n" \
                     f"1. Define the function signature\n" \
                     f"{TOFILL_TOKEN}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "steps-specific":
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you summarize this {src_lang} program into a few steps such that a reader can easily rewrite the program in {tgt_lang}?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### {tgt_lang} steps\n\n" \
                     f"1. Define the function signature\n" \
                     f"{TOFILL_TOKEN}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "steps-nl":
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you summarize this {src_lang} program into a few steps such that a reader can easily rewrite the program in {tgt_lang}?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### {tgt_lang} steps\n\n" \
                     f"1. Define the function signature\n" \
                     f"2. {TOFILL_TOKEN}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "steps-w-code":
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you summarize this {src_lang} program into a few steps such that a reader can easily rewrite the program in {tgt_lang}?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### {tgt_lang} steps\n\n" \
                     f"1. Define the function signature\n" \
                     f"```{tgt_lang.lower()}\n" \
                     f"{tgt_sig_line.strip()}\n" \
                     f"{TOFILL_TOKEN}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "steps-latex":
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you summarize this {src_lang} program into a few steps in English and then Latex pseudocode such that a reader can easily rewrite the program in {tgt_lang}?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### {tgt_lang} steps\n\n" \
                     f"1. Define the function signature\n" \
                     f"{TOFILL_TOKEN}\n\n" \
                     f"### Latex pseudocode\n\n" \
                     "\\begin{algorithm}\n" \
                     f"{TOFILL_TOKEN}\n" \
                     "\\end{algorithm}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "comments":
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you add inline comments to this {src_lang} program such that a reader can easily rewrite the program in {tgt_lang}?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### Commented {src_lang} version\n\n" \
                     f"{get_python_signature_from_code(src_code)}\n" \
                     f"{TOFILL_TOKEN}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "align":
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Please translate the following {src_code} code to {tgt_lang}? While translating, make sure to indicate which section of the original {src_code} code the translated statement come from and explain what the section of the code means.\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "latex":
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you summarize this {src_lang} program into a few steps in Latex pseudocode such that a reader can easily rewrite the program in {tgt_lang}?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### Latex pseudocode\n\n" \
                     "\\begin{algorithm}\n" \
                     f"{TOFILL_TOKEN}\n" \
                     "\\end{algorithm}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "explain":
            multi_view_str = extract_multi_views(multi_view_files, src_lang, tgt_lang, no_header=True, no_code=True)
            exp = multi_view_str if multi_view_files else f"This{TOFILL_TOKEN}"
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you explain what this {src_lang} program does with a couple of sentences? The goal with the explanation, is so that a reader can easily rewrite the program in {tgt_lang}.\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### Explanation\n\n" \
                     f"{exp}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "explain-lbl":
            multi_view_str = extract_multi_views(multi_view_files, src_lang, tgt_lang, no_header=True, no_code=True)
            exp = multi_view_str if multi_view_files else f"The code is an implementation of{TOFILL_TOKEN}"
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you explain what this {src_lang} program does line by line? The goal with the explanation, is so that a reader can easily rewrite the program in {tgt_lang}.\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### Explanation\n\n" \
                     f"{exp}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "explain-lbl-simp":
            multi_view_str = extract_multi_views(multi_view_files, src_lang, tgt_lang, no_header=True, no_code=True)
            exp = multi_view_str if multi_view_files else f"The code is an implementation of{TOFILL_TOKEN}"
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you explain what this {src_lang} program does line by line? If a line is too long or too complicated, simplify it and explain what " \
                     f"individual parts of the line mean first before explaining the whole line. The goal with the explanation, is so that a " \
                     f"reader can easily rewrite the program in {tgt_lang}.\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### Explanation\n\n" \
                     f"{exp}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "explain-lbl-simp-trans":
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you explain what this {src_lang} program does line by line? If a line is too long or too complicated, simplify it and explain what " \
                     f"individual parts of the line mean first before explaining the whole line. After explaining each line, provide a possible translation of that line in {tgt_lang}. " \
                     f"The goal with the explanation, is so that a reader can easily rewrite the program in {tgt_lang}.\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### Explanation\n\n" \
                     f"The code is an implementation of{TOFILL_TOKEN}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "summary":
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you summarize this {src_lang} program in a few sentences such that a reader can easily rewrite the program in {tgt_lang}?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### Summary\n\n" \
                     f"This {TOFILL_TOKEN}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "summary-llm":
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you summarize this Python program in English such that it is easier to translate (to {tgt_lang}) for a large language model?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### Summary\n\n" \
                     f"This {TOFILL_TOKEN}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "gold-summary":  # Not actually multiturn, but putting it here for now
            assert tgt_lang.lower() == "java", "Imeplementation only works for Java for now"
            docstr, tgt_sig_all = get_and_remove_java_docstring_from_prompt(tgt_sig_all)
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you summarize this {src_lang} program in a few sentences such that a reader can easily rewrite the program in {tgt_lang}?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### Summary\n\n" \
                     f"{docstr}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "summary-steps":
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you summarize this {src_lang} program in a few sentences, then in a few English steps such that a reader can easily rewrite the program in {tgt_lang}?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### Summary\n\n" \
                     f"This function{TOFILL_TOKEN}\n\n" \
                     f"### {tgt_lang} steps\n\n" \
                     f"1. Define the function signature\n" \
                     f"{TOFILL_TOKEN}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "rewrite":
            assert src_lang == "Python"
            src_sig = src_code[src_code.find("def"): src_code.find("\n", src_code.find("def"))]
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you rewrite or simplify this {src_lang} program so that it can be translated to {tgt_lang} more easily?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### Rewritten {src_lang} version\n\n" \
                     f"{src_sig}\n" \
                     f"{TOFILL_TOKEN}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "cot":
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you translate this {src_lang} program to {tgt_lang}? We can think step by step on translating each sub-components of the programs.\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### Step by step\n\n" \
                     f"Let's think line by line on how we can translate the above function:\n\n" \
                     f"{src_lang}: {get_python_signature_from_code(src_code)}\n" \
                     f"{tgt_lang}: {tgt_sig_line.strip()}\n\n" \
                     f"{src_lang}: {TOFILL_TOKEN}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "cot-reasoning":
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you step by step reason through how you would translate this {src_lang} program into {tgt_lang}?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### Step by step reasoning\n\n" \
                     f"{TOFILL_TOKEN}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "steps-cot":
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                 f"Can you summarize this {src_lang} program into a few steps such that a reader can easily rewrite the program in other " \
                 f"languages?\n\n" \
                 f"### {src_lang} version\n\n" \
                 f"{src_code}" \
                 f"### steps\n\n" \
                 f"1. Define the function signature\n\n" \
                 f"{TOFILL_TOKEN}\n" \
                 f"Give above steps, can you write a {tgt_lang} program that does the same thing as the {src_lang} program provided? Let's think step by step on how we can translate each step:\n\n" \
                 f"step 1: Define translated function signature:\n\n" \
                 f"{tgt_sig_line}\n\n" \
                 f"step 2:{TOFILL_TOKEN}\n\n" \
                 f"### {tgt_lang} version\n\n" \
                 f"{tgt_sig_all}"
        elif multiturn_prompt == "multi-view":
            assert multi_view_files is not None
            multi_view_str = extract_multi_views(multi_view_files, src_lang, tgt_lang)
            # prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
            #          f"Here are several pieces of information (code, natural language description, pseudocode, etc.) describing the same {src_lang} program. Can you rewrite this program in {tgt_lang}?\n\n" \
            #          f"### {src_lang} version\n\n" \
            #          f"{src_code}" \
            #          f"{multi_view_str}\n\n" \
            #          f"### {tgt_lang} version\n\n" \
            #          f"{tgt_sig_all}"
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you explain what this {src_lang} program does with a couple of sentences? The goal with the explanation, is so that a reader can easily rewrite the program in {tgt_lang}.\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"{multi_view_str}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt.startswith("pivot-"):
            _, correct_program, pivot_lang = multiturn_prompt.split("-")
            pivot_lang_print = cap(SHORT2CANONICAL[pivot_lang])
            pivot_translator = __import__(f"humaneval_to_{pivot_lang}").Translator()
            if correct_program == "gold":
                pivot_program = get_gold_program(pivot_translator, problem["original"].split("/")[-2], pivot_lang, problem["name"])
            elif correct_program == "wrong":
                wrong_programs = get_wrong_programs(problem["original"].split("/")[-2], pivot_lang)
                pivot_program = sample_random_wrong_program(pivot_translator, wrong_programs, problem["name"])
                pivot_program = pivot_program[0] if pivot_program is not None else pivot_program
            else:
                raise NotImplementedError("pivot experiment needs to be in the form of pivot-{gold/wrong}-{pivot_lang}")
            if pivot_program is None:
                return None
            prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                     f"Can you translate this {src_lang} program to {pivot_lang_print} and then translate it to {tgt_lang}?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### {pivot_lang_print} version\n\n" \
                     f"{pivot_program.strip()}\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "func":
            assert src_lang == "Python"
            func_calls = extract_python_function_calls(src_code)
            if len(func_calls) == 0:
                prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                         f"Can you translate this {src_lang} program to {tgt_lang}?\n\n" \
                         f"### {src_lang} version\n\n" \
                         f"{src_code}" \
                         f"### {tgt_lang} version\n\n" \
                         f"{tgt_sig_all}"
            else:
                if len(func_calls) == 1:
                    method_call_str = f"The {src_lang} program contains the following method call that need to be translated: `{func_calls[0]}`. " \
                                      f"Let's translate this and pay attention to each input variable type " \
                                      f"and the context in which the method call is found."
                else:
                    func_calls_str = ", ".join([f"`{c}`" for c in func_calls[:-1]]) + f", and `{func_calls[-1]}`"
                    method_call_str = f"The {src_lang} program contains the following method calls that need to be translated: {func_calls_str}. " \
                                      f"Let's translate them one by one and pay attention to each input variable type " \
                                      f"and the context in which the method calls are found."
                prompt = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                         f"Can you translate this {src_lang} program to {tgt_lang}? Before translating the entire " \
                         f"program, let's think step by step on how to translate each {src_lang} method calls.\n\n" \
                         f"### {src_lang} version\n\n" \
                         f"{src_code}" \
                         f"### Method call translation\n\n" \
                         f"{method_call_str}\n" \
                         f"- {TOFILL_TOKEN}\n\n" \
                         f"### {tgt_lang} version\n\n" \
                         f"{tgt_sig_all}"
        elif multiturn_prompt == "func2":
            prompt = f"Given the following {src_lang} program, can you extract a numbered list of native {src_lang} functions used?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### {src_lang} native functions\n\n" \
                     f"Certainly! Here is the list of native {src_lang} functions used in the given program:\n" \
                     f"1. {TOFILL_TOKEN}\n\n" \
                     f"For each of the above functions, can you give the corresponding {tgt_lang} translation?\n\n" \
                     f"### {tgt_lang} individual function translation\n\n" \
                     f"Certainly! Here are the corresponding {tgt_lang} translations for each of the {src_lang} functions used in the given program:\n" \
                     f"1. {TOFILL_TOKEN}\n\n" \
                     f"Now can you translate the original {src_lang} program to {tgt_lang}? You should use comments to indicate that you are using the corresponding {tgt_lang} translations that you generated previously.\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "func3":
            prompt = f"Given the following {src_lang} program, can you extract a numbered list of native {src_lang} functions used?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"### {src_lang} native functions\n\n" \
                     f"{TOFILL_TOKEN}\n\n" \
                     f"For each of the above functions, can you give the corresponding {tgt_lang} translation?\n\n" \
                     f"### {tgt_lang} individual function translation\n\n" \
                     f"{TOFILL_TOKEN}\n\n" \
                     f"Now can you translate the original {src_lang} program to {tgt_lang}? You should use comments to indicate that you are using the corresponding {tgt_lang} translations that you generated previously.\n\n" \
                     f"### {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"
        elif multiturn_prompt == "debug":
            multi_view_str = extract_multi_views(multi_view_files, src_lang, src_lang)
            end_program = "\n}" if "\n}" in problem["stop_tokens"] else ""
            prompt = f"You are a helpful and faithful compiler that transpiles {src_lang} code to {tgt_lang} code. Please translate the following {src_lang} code to {tgt_lang}?\n\n" \
                     f"### {src_lang} version\n\n" \
                     f"{src_code}" \
                     f"{multi_view_str}{end_program}\n\n" \
                     f"The above {tgt_lang} translation does not do the same thing as the {src_lang} code. Correct the {tgt_lang} translation.\n\n" \
                     f"### Revised {tgt_lang} version\n\n" \
                     f"{tgt_sig_all}"

        else:
            raise NotImplementedError
    else:
        prompt = [
            {
                "role": "system",
                "content": "You are an helpful AI assistant who understands all programming languages and can translate between them at ease."
            },
            {
                "role": "user",
                "content": f"Can you summarize this {src_lang} program into a few steps such that a reader can easily rewrite the program in other languages?\n\n### {src_lang} version\n\n{src_code}### steps\n\n1. define the function signature"
            },
            {
                "role": "assistant",
                "content": None
            }
        ]

        if multiturn_prompt == "steps":
            prompt.append({
                "role": "user",
                "content": f"Can you complete the following {tgt_lang} program with above steps? Keep the {tgt_lang} signature exactly the same as the code provided.\n\n### {tgt_lang} version\n\n{tgt_sig_all}"
            })
        elif multiturn_prompt == "steps-cot":
            # TODO doesn't work as well on Java
            prompt.extend([
                {
                    "role": "user",
                    "content": f"Give above steps, can you write a {tgt_lang} program that does the same thing as the {src_lang} program provided? Let's think step by step on how we can translate each step:\n\nstep 1: define translated function signature:\n\n{tgt_sig_line}\n\nstep 2:"
                },
                {
                    "role": "assistant",
                    "content": None
                },
                {
                    "role": "user",
                    "content": f"Can you complete the following {tgt_lang} program with above steps? Keep the {tgt_lang} signature exactly the same as the code provided.\n\n### {tgt_lang} version\n\n{tgt_sig_all}"
                }
            ])
        else:
            raise NotImplementedError

    # add few shot examples in
    if len(few_shots) > 0 and shots > 0:
        if isinstance(prompt, str) and isinstance(few_shots, str):
            turns_per_shot = 3 if multiturn_prompt == "func" else len(list(find_all(prompt, "### ")))
            fs_indices = list(find_all(few_shots, "### "))
            turns_in_few_shots = len(fs_indices)
            assert turns_in_few_shots % turns_per_shot == 0 and (turns_in_few_shots / turns_per_shot) >= shots, \
                "multi-turn few shot file is not the format we expected it to be in"

            # first get rid of task instructions
            prompt = prompt[prompt.find("###"):]
            fs_indices.append(-1)
            few_shot_str = few_shots[:fs_indices[turns_per_shot * shots]].strip()
            if any(f in src_code for f in [
                # "is_sorted",
                "even_odd_palindrome",
                # "separate_paren_groups",
                # "sum_squares"
            ]):
                x=1
            if src_code.replace(" \n","\n").strip() in few_shot_str.replace(" \n","\n").strip():
                return None
            prompt = few_shot_str + "\n\n" + prompt
        elif isinstance(few_shots, pd.Series):
            assert multiturn_prompt == "summary" and isinstance(prompt, str)
            few_shot_str = f"You are an helpful AI assistant who understands all programming languages and can translate between them at ease. " \
                           f"Can you summarize this {src_lang} program in a few sentences such that a reader can easily rewrite the program in {tgt_lang}?\n\n"
            for i in range(1, shots + 1):
                few_shot_str += "### Python version\n\n" \
                            f"{few_shots['top' + str(i) + '_src']}" \
                            f"### Summary\n\n" \
                            f"{few_shots['top' + str(i) + '_summary']}" \
                            f"### {tgt_lang} version\n\n" \
                            f"{few_shots['top' + str(i) + '_tgt']}\n}}\n\n"
            prompt = few_shot_str + prompt[prompt.find("###"):]
    else:
            turns_per_shot = len(prompt)
            assert len(few_shots) % turns_per_shot == 0 and (len(few_shots) / turns_per_shot) >= shots, \
                "multi-turn few shot file is not the format we expected it to be in"
            for i in range(shots):
                cur_shot = few_shots[i*turns_per_shot: (i+1)*turns_per_shot]
                prompt = [prompt[0], *cur_shot, *prompt[1:]]

    problem["prompt"] = tgt_sig_all
    if multiturn_template_count > 1:
        if TOFILL_TOKEN not in prompt:
            print(f"Multi-turn template count chosen to be {multiturn_template_count}, but does not need to be since "
                  f"promot is not multi-turn in nature")
        else:
            prompt = [prompt] * multiturn_template_count
    problem["translation_prompt"] = prompt
    return problem
