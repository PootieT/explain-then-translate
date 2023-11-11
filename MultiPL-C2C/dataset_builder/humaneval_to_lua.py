# This script translates problems from the OpenAI HumanEval dataset into Lua.
import re
import ast
from typing import List

from pygments import lex
from pygments.lexers import get_lexer_by_name

from base_language_translator import LanguageTranslator

# We turn multi-line docstrings into single-line comments. This captures the
# start of the line.
from dataset_builder.utils import fix_indentation, remove_pre_function_comments, remove_trailing_comments, find_all, \
    remove_tests

DOCSTRING_LINESTART_RE = re.compile("""\n(\s+)""")

TargetExp = str

class Translator(LanguageTranslator[TargetExp]):

    def __init__(self):
        self._lexer = None

    @property
    def lexer(self):
        if self._lexer is None:
            self._lexer = get_lexer_by_name("lua")
        return self._lexer

    def tokenize_code(self, script):
        return [t[1].strip() for t in lex(script, self.lexer)]


    def stop(self):
        return [
            # regular completion stops
            # "\nlocal",
            # "\nfunction",
            "\n--",
            # "\n\n"
            "\n#",  # from me
            "\nend",  # from me
            # chat completion stops
            # "\n-- Test",
            # "\n-- test",
            # "\n-- example",
            # "\n-- Example",
            # "###" # doesn't work as well
        ]

    def file_ext(self) -> str:
        return "lua"

    def module_imports(self) -> str:
        return "\n".join([
            "math = require('math')",
        ]) + "\n"

    def translate_prompt(self, name: str, args: List[ast.arg], _returns: ast.expr, description: str) -> str:
        lua_description = (
            "-- " + re.sub(DOCSTRING_LINESTART_RE, "\n-- ", description.strip()) + "\n"
        ) if len(description) > 0 else ""
        arg_names = [arg.arg for arg in args]
        arg_list = ", ".join(arg_names)
        return f"{self.module_imports()}{lua_description}local function {name}({arg_list})\n"

    def test_suite_prefix_lines(self, entry_point: str) -> List[str]:
        """
        This code goes at the start of the test suite.
        """
        return [
            "end",
            "",
            "lu = require('luaunit')",
            "",
            "function test_humaneval()",
            f"local candidate = {entry_point}",
        ]

    def test_suite_suffix_lines(self) -> List[str]:
        return ["end", "", "os.exit(lu.LuaUnit.run())"]

    def deep_equality(self, left: TargetExp, right: TargetExp) -> str:
        """
        All tests are assertions that compare deep equality between left and right.

        Make sure you use the right equality operator for your language. For example,
        == is the wrong operator for Java and OCaml.
        """
        return "    lu.assertEquals({}, {})".format(left, right)

    def gen_literal(self, c: bool | str | int | float | None) -> TargetExp:
        """Translate a literal expression
        c: is the literal value
        """
        if type(c) == bool:
            return str(c).lower()
        return repr(c)

    def gen_var(self, v: str) -> TargetExp:
        """Translate a variable with name v."""
        return v

    def gen_list(self, l: List[TargetExp]) -> TargetExp:
        """Translate a list with elements l
        A list [ x, y, z] translates to { x, y, z }
        """
        return "{" + ", ".join(l) + "}"

    def gen_tuple(self, t: List[TargetExp]) -> TargetExp:
        """Translate a tuple with elements t
        A tuple (x, y, z) translates to { x, y, z }
        """
        return "{" + ", ".join(t) + "}"

    def gen_dict(self, keys: List[TargetExp], values: List[TargetExp]) -> TargetExp:
        """Translate a dictionary with keys and values
        A dictionary { "key1": val1, "key2": val2 } translates to { ["key1"] = val1, ["key2"] = val2 }
        """
        return "{" + ", ".join(f"[{k}] = {v}" for k, v in zip(keys, values)) + "}"

    def gen_call(self, func: TargetExp, args: List[TargetExp]) -> TargetExp:
        """Translate a function call `func(args)`
        A function call f(x, y, z) translates to f(x, y, z)
        """
        return func + "(" + ", ".join(args) + ")"

    def get_function_name(self, line: str):
        assert self.is_function_signature(line)
        return line.split("(")[0].split()[-1].strip()

    @staticmethod
    def is_function_signature(line: str):
        return line.strip().startswith("function ") or line.strip().startswith("local function ")

    def extract_functions(self, script: str) -> List[str]:
        # copied from Julia
        def same_level_end(script_lines: List[str]) -> bool:
            indent = "    "
            indent_func = script_lines[0].replace(script_lines[0].strip(), "").count(indent)
            indent_end = script_lines[-1].replace(script_lines[-1].strip(), "").count(indent)
            return indent_end == indent_func

        functions = []
        curr_func = None
        for l in script.split("\n"):
            if self.is_function_signature(l):
                if curr_func is None:
                    curr_func = [l]
                else:
                    # Lua allows nested functions. but we just treat
                    # this as part of the parent function.
                    curr_func.append(l)
            else:
                if curr_func is not None:
                    curr_func.append(l)

            # still a naive implementation. Indentation is not essential in Lua syntax, which means
            # checking indentation of end is not necessarily the formal way to do it. but otherwise
            # we will have to check all the loops, a lot more semantics involved
            if curr_func is not None and curr_func[-1].strip().startswith("end") and same_level_end(curr_func):
                # function completed!
                functions.append("\n".join(curr_func))
                curr_func = None
        return functions

    def remove_imports(self, script: str):
        # only remove first n continuous lines of imports
        script_lines = script.split("\n")
        i = 0
        for i, line in enumerate(script_lines):
            if not any([w in set(self.tokenize_code(line)) for w in ["require", "module"]]):
                break
        return "\n".join(script_lines[i:])


    @staticmethod
    def remove_class_decl(script):
        return script

    @staticmethod
    def is_comment(line: str):
        return line.strip().startswith("--")

    def completion_touchup(self, script: str, prompt: str):
        if "### lua" in script.lower():
            script = "\n".join([l for l in script.split("\n") if not l.strip().lower().startswith("### lua")])
        # if generation is not our desired complete function, fix the indentation
        main_func_name = self.get_function_name(prompt.strip().split("\n")[-1])

        # remove preceeding comments to before class/function definition
        script = remove_pre_function_comments(script, self.is_function_signature, self.is_comment)

        # if last return statement is not followed by an end, add an end at the next line
        # return_indices = list(find_all(script, "return"))
        # if "end" not in script[return_indices[-1]:]:
        #     return_eol = script.find("\n", return_indices[-1])+1
        #     script = script[:return_eol] + "\nend" + script[return_eol:]

        # remove all tokens after the last end
        script = remove_trailing_comments(script, "end", is_comments=self.is_comment)
        script = remove_trailing_comments(script, "end")
        if main_func_name not in script.split("\n")[0]:
            script = fix_indentation(script, is_comment=self.is_comment, indentation_level=1)
        return script

    @staticmethod
    def remove_tests(script: str):
        return remove_tests(script, "lu = require('luaunit')", "")


if __name__ == "__main__":
#     t = Translator()
#     s = """local math = require("math")
#
# local function max_fill(grid, capacity)
#     local total = 0
#     for i = 1, #grid do
#         local sum = 0
#         for j = 1, #grid[i] do
#             sum = sum + grid[i][j]
#         end
#         total = total + math.ceil(sum / capacity)
#     end
#     return total
# end
# """
# #     # a = fix_indentation(s, t.is_comment, indentation_level=0)
#     a = t.remove_imports(s)
#     a = t.completion_touchup(s, "local function max_fill(grid, capacity)")
# #     a = t.extract_functions(s)
#     print(a)
    pass
