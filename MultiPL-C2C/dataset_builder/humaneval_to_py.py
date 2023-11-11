# This script translates problems from the OpenAI HumanEval dataset into Python.
# It may seem silly, since the HumanEval dataset is already in Python. But,
# we use this script uniformly across all languages to translate the dataset
# in various ways.
import ast
from string import ascii_lowercase
from typing import List

from base_language_translator import LanguageTranslator
from codegen_sources.model.src.utils import TREE_SITTER_ROOT
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor
from dataset_builder.utils import remove_pre_function_comments, fix_indentation, remove_tests


def translate_type(t, needs):
    match t:
        case ast.Subscript(ast.Name(id), ast.Tuple(elts), ctx):
            needs(id)
            tys = [translate_type(elem, needs) for elem in elts]
            return id + "[" + ", ".join(tys) + "]"
        case ast.Subscript(ast.Name(c), of):
            needs(c)
            return c + "[" + translate_type(of, needs) + "]"
        case ast.Name("Any" as x):
            needs(x)
            return x
        case ast.Name(x):
            return x
        case "str" | "float":
            return t
        case ast.Constant(None):
            return "None"
        case ast.Constant(Ellipsis):
            return "..."
        case ast.Constant(x):
            print(f"unknown constant {x} {type(x)}")
        case _other:
            print("other")
            raise Exception(f"unknown annotation: {t}")

TargetExp = str

class Translator(LanguageTranslator[TargetExp]):

    @property
    def lang_processor(self):
        if not hasattr(self, "_lang_processor"):
            self._lang_processor = LangProcessor.processors["python"](root_folder=TREE_SITTER_ROOT)
        return self._lang_processor

    def stop(self):
        return ["\ndef", "\n#", "\nif", "\nclass"]

    @property
    def additional_stops(self):
        return [
            "\n*",
            "\n`",
            *[f"\n{i}" for i in ascii_lowercase],
        ]

    def file_ext(self):
        return "py"

    def translate_prompt(self, name: str, args: List[ast.arg], returns, description: str) -> str:
        py_description = "\"\"\"" + description + "\"\"\"" if description else ""
        needs = []
        add_need = lambda x: needs.append(x) if not x in needs else None
        def name_and_type(arg):
            if arg.annotation is None:
                return arg.arg
            else:
                return f"{arg.arg}: {translate_type(arg.annotation, add_need)}"
        arg_list = ", ".join([name_and_type(arg) for arg in args])
        ann = "" if returns is None else f" -> {translate_type(returns, add_need)}"
        imports = "" if needs == [] else f"from typing import {', '.join(needs)}\n\n"
        return f"{imports}def {name}({arg_list}){ann}:\n    {py_description}\n"

    def test_suite_prefix_lines(self, entry_point) -> List[str]:
        """
        This code goes at the start of the test suite.
        """
        self.entry_point = entry_point
        return [
            "def check(candidate):",
        ]

    def test_suite_suffix_lines(self) -> List[str]:
        return [
            "",
            "def test_check():",
            f"    check({self.entry_point})",
            "",
            "test_check()",
            "",
        ]

    def deep_equality(self, left: str, right: str) -> str:
        return f"    assert {left} == {right}"

    def gen_literal(self, c: bool | str | int | float | None):
        return repr(c)

    def gen_var(self, v: str) -> str:
        return v

    def gen_list(self, l: List[str]) -> str:
        return "[" + ", ".join(l) + "]"

    def gen_tuple(self, t: List[str]) -> str:
        trailing = "," if len(t) == 1 else ""
        return "(" + ", ".join(t) + trailing + ")"

    def gen_dict(self, keys: List[str], values: List[str]) -> str:
        return "{ " + ", ".join(f'{k}: {v}' for k, v in zip(keys, values)) + " }"

    def gen_call(self, func: str, args: List[str]) -> str:
        return func + "(" + ", ".join(args) + ")"

    def get_function_name(self, line: str):
        assert self.is_function_signature(line)
        return line.split("(")[0].split()[-1].strip()

    def is_function_signature(self, line: str):
        return line.strip().startswith("def")

    def extract_original_functions(self, script, detokenized_functions: List[str]):
        if len(detokenized_functions) == 0:
            return []
        script_lines = script.split("\n")
        functions = []
        for detokenized_func in detokenized_functions:
            func_name = self.get_function_name(detokenized_func)
            for i, line in enumerate(script_lines):
                if self.is_function_signature(line) and func_name in self.lang_processor.tokenize_code(line):
                    func_lines = script_lines[i: i+len(detokenized_func.strip().split("\n"))]
                    function = remove_trailing_comments("\n".join(func_lines))
                    functions.append(function)
                    break
        assert len(functions) == len(detokenized_functions)
        return functions

    def extract_functions(self, script: str) -> List[str]:
        # transcoder parser is trash
        try:
            tokenized_code = " ".join(self.lang_processor.tokenize_code(script, keep_comments=True))
            tokenized_functions, _ = self.lang_processor.extract_functions(tokenized_code)
        except:
            tokenized_functions = []
        detokenized_functions = [self.lang_processor.detokenize_code(f) for f in tokenized_functions]
        functions = self.extract_original_functions(script, detokenized_functions)
        return functions

    @staticmethod
    def remove_imports(script: str):
        return "\n".join([l for l in script.split("\n") if not "import" in l.split()])

    @staticmethod
    def remove_class_decl(script):
        return script

    @staticmethod
    def is_comment(line: str):
        return line.startswith("#")

    def completion_touchup(self, script:str, prompt: str):
        script = fix_indentation(script, self.is_comment, indentation_level=1)
        # remove preceeding comments to before class definition
        script = remove_pre_function_comments(script, self.is_function_signature, self.is_comment)

        # if completion is only comments (generate comment step), add back original source code
        if "return" not in script:
            script = self.add_source_program(script, prompt)
        return script

    def add_source_program(self, script, prompt):
        original_program = prompt[prompt.find("### original python code"):prompt.find("### commented python code")]
        script = script + original_program[original_program.find("\n", original_program.find("def")):]
        return script

    def remove_trailing_comments(self, script):
        script_lines = script.split("\n")
        # remove bottom lines as long as that line is comment
        while len(script_lines) > 0 and (script_lines[-1].strip().startswith("#") or not script_lines[-1].strip()):
            script_lines = script_lines[:-1]
        script = "\n".join(script_lines)
        return script

    @staticmethod
    def remove_tests(script: str):
        return remove_tests(script, "def check(candidate):", "")


if __name__=="__main__":
    # t = Translator()
    # s0 = '# Import the math module to use the ceil function\n    \n    # Initialize a variable to store the sum of squares\n    squared = 0\n    \n    # Loop through the list of floats\n    for i in lst:\n        \n        # Calculate the square of the ceiling of each float and add it to the sum\n        squared += math.ceil(i)**2\n    \n    # Return the sum of squares as an integer\n    return squared'
    # o0 = t.fix_indentation(s0)
    # a0 = '    # Import the math module to use the ceil function\n    \n    # Initialize a variable to store the sum of squares\n    squared = 0\n    \n    # Loop through the list of floats\n    for i in lst:\n        \n        # Calculate the square of the ceiling of each float and add it to the sum\n        squared += math.ceil(i)**2\n    \n    # Return the sum of squares as an integer\n    return squared'
    # assert a0 == o0
    # s1 = '# Import the math module to use the ceil function\n    \n# Initialize a variable to store the sum of squares\nsquared = 0\n\n# Loop through the list of floats\nfor i in lst:\n    \n    # Calculate the square of the ceiling of each float and add it to the sum\n    squared += math.ceil(i)**2\n\n# Return the sum of squares as an integer\nreturn squared'
    # o1 = t.fix_indentation(s1)
    # a1 = '    # Import the math module to use the ceil function\n    \n    # Initialize a variable to store the sum of squares\n    squared = 0\n    \n    # Loop through the list of floats\n    for i in lst:\n        \n        # Calculate the square of the ceiling of each float and add it to the sum\n        squared += math.ceil(i)**2\n    \n    # Return the sum of squares as an integer\n    return squared'
    # assert o1 == a1
    pass