# This script translates problems from the OpenAI HumanEval dataset into Julia.
# 
# Julia is a very similar language to python, therefore, most of the translation 
# is just between various keywords. Julia includes support for arbitrary Union 
# types (with Optional being represented as Union{T, Nothing}). Since Julia is 
# (by and large) dynamically typed, no coercions or type inference are necessary 
# to generate benchmarks. 
#
# Julia can be installed by running the script from jill.sh: 
# wget https://raw.githubusercontent.com/abelsiqueira/jill/main/jill.sh
# 
# The script can both be run as root or user.
import re
import ast
from typing import List, Optional

# We turn multi-line docstrings into single-line comments. This captures the
# start of the line.
from dataset_builder.utils import remove_trailing_comments, remove_pre_function_comments, fix_indentation, remove_tests

DOCSTRING_LINESTART_RE = re.compile("""\n(\s+)""")

# i don't want this to be global but i also don't want to pass it all the way
# through translate_type
needs_hashmap = False

def translate_type(t):
    global needs_hashmap
    match t:
        case ast.Subscript(ast.Name(id), slice, ctx):
            match id:
                case "List":
                    return "Vector{" + translate_type(slice) + "}"
                case "Union":
                    match slice: 
                        case ast.Tuple(elts, _ctx): 
                            tys = [translate_type(elem) for elem in elts]
                            return "Union{" + ", ".join(tys) + "}"
                        case other: 
                            raise Exception(f"Unexpected slice: {slice}")
                case "Tuple":
                    match slice:
                        case ast.Tuple(elts, _ctx):
                            tys = [translate_type(elem) for elem in elts]
                            return "Tuple{" + ", ".join(tys) + "}"
                        case other:
                            raise Exception(f"Bad tuple: {slice}")
                case "Dict":
                    match slice:
                        case ast.Tuple([ast.Name(k), ast.Name(v)], _ctx):
                            key, value = translate_type(k), translate_type(v)
                            needs_hashmap = True
                            return f"Dict{{{key}, {value}}}>"
                        case other:
                            raise Exception(f"Bad dict: {slice}")
                case "Optional":
                    return "Union{" + translate_type(slice) + ", Nothing}"
                case other:
                    raise Exception(f"Bad generic {other}")
        case ast.Name("int") | "int":
            return "Int64"
        case ast.Name("float"):
            return "Float64"
        case ast.Name("bool"):
            return "Bool"
        case ast.Name("str") | "str":
            return "String"
        case None:
            raise Exception("implicitly untyped argument")
        case ast.Name("Any"):
            return "Any"
        case ast.Name(x):
            raise Exception(f"unknown name {x}")
        case ast.Constant(Ellipsis):
            raise Exception("no ellipsis!!")
        case _other:
            raise Exception(f"unknown annotation: {t}")

def coerce(expr: str, type) -> str: 
    match expr, type:
        case "[]", ast.Subscript(ast.Name("List"), to): 
            return f"Vector{{{translate_type(to)}}}([])"
        case _: 
            return expr


class Translator:

    stop = [
        # regular completion
        # "\nfunction",  ## cannot be used with multi step translations
        # "\nmacro",
        # "\n\n",
        "\n#",
        "\nend",  # this is a more strict way of ending completion, chatgpt never returns multiple ends at base indentation level
        # chat completion
        # "\nmacro"
    ]

    def __init__(self):
        global needs_hashmap
        self.type = None
        self.is_candidate_result = False

    def file_ext(self):
        return "jl"

    def translate_prompt(self, name: str, args: List[ast.arg], returns, description: str) -> Optional[str]:
        self.type = [[arg.annotation for arg in args], returns]
        def translate_arg(arg):
            return arg.arg + "::" + translate_type(arg.annotation)
        arg_strings = []
        return_type = ""
        description = f"\"\"\"{description}\"\"\"\n" if len(description) > 0 else ""
        try:
            arg_strings = [translate_arg(arg) for arg in args]
            return_type = translate_type(returns)
        except Exception as e:
            print(e)
            return None
        arg_list = ", ".join(arg_strings)
        return f"{description}function {name}({arg_list})::{return_type} \n"

    def test_suite_prefix_lines(self, entry_point) -> List[str]:
        """
        This code goes at the start of the test suite.
        """
        return [
            "end",  # added this and line below since we use end function as stop token
            "",
            "using Test",
            "",
            "@testset begin",
            "",
            f"candidate = {entry_point};",
        ]

    def test_suite_suffix_lines(self) -> List[str]:
        return ["end\n"]

    def deep_equality(self, left: str, right: str) -> str:
        """
        All tests are assertions that compare deep equality between left and right.

        Make sure you use the right equality operator for your language. For example,
        == is the wrong operator for Java and OCaml.
        """
        if self.is_candidate_result:
            right = coerce(right, self.type[1])
            self.is_candidate_result = False
        return f"\t@test({left} == {right})"

    def gen_literal(self, c: bool | str | int | float):
        """Translate a literal expression
        c: is the literal value
        """
        if type(c) == bool:
            return str(c).lower()
        if type(c) == str:
            escaped = c.translate(str.maketrans({
                '"': r"\"",
                "$": r"\$"
            }))
            return '"' + escaped + '"'
        if c is None: 
            return "nothing"
        return repr(c)

    def gen_unaryop(self, op: str, v: str) -> str:
        """Translate a unary operation (op, v)"""
        return op + v

    def gen_var(self, v: str) -> str:
        """Translate a variable with name v."""
        return v

    def gen_list(self, l: List[str]) -> str:
        """Translate a list with elements l
        A list [ x, y, z] translates to { x, y, z }
        """
        return "[" + ", ".join(l) + "]"

    def gen_tuple(self, t: List[str]) -> str:
        """Translate a tuple with elements t
        A tuple (x, y, z) translates to { x, y, z }
        """
        return "(" + ", ".join(t) + ")"

    def gen_dict(self, keys: List[str], values: List[str]) -> str:
        """Translate a dictionary with keys and values
        A dictionary { "key1": val1, "key2": val2 } translates to { ["key1"] = val1, ["key2"] = val2 }
        """
        return "Dict(" + ", ".join(f"{k} => {v}" for k, v in zip(keys, values)) + ")"

    def gen_call(self, func: str, args: List[str]) -> str:
        """Translate a function call `func(args)`
        A function call f(x, y, z) translates to f(x, y, z)
        """
        if func == "candidate": 
            self.is_candidate_result = True
            args = [coerce(arg, self.type[0][i]) for i, arg in enumerate(args)]
        return func + "(" + ", ".join(args) + ")"

    def get_function_name(self, line: str):
        assert self.is_function_signature(line)
        return line.split("(")[0].split()[-1].strip()

    def is_function_signature(self, line: str):
        return line.strip().startswith("function ")

    def extract_functions(self, script: str) -> List[str]:
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
                    # Julia allows nested functions. but we just treat
                    # this as part of the parent function.
                    curr_func.append(l)
            else:
                if curr_func is not None:
                    curr_func.append(l)

            # TODO need to make sure baseline indentation level is correct
            if curr_func is not None and curr_func[-1].strip().startswith("end") and same_level_end(curr_func):
                # function completed!
                functions.append("\n".join(curr_func))
                curr_func = None
        return functions

    @staticmethod
    def remove_imports(script: str):
        return "\n".join([l for l in script.split("\n") if not \
               any([l.strip().startswith(w) for w in ["import", "using", "include"]])])

    @staticmethod
    def remove_class_decl(script):
        return script

    @staticmethod
    def is_comment(line: str):
        return line.strip().startswith("#")

    def completion_touchup(self, script: str, prompt: str):
        script = fix_indentation(script, self.is_comment, indentation_level=0)

        # remove preceeding comments to before class/function definition
        script = remove_pre_function_comments(script, self.is_function_signature, self.is_comment)

        script = remove_trailing_comments(script, "end")
        return script

    @staticmethod
    def remove_tests(script: str):
        return remove_tests(script, "using Test", "")


if __name__=="__main__":
    # t = Translator()
    # s = "function add_one(i)    check = split(txt, " ")[end]\n    return length(check) == 1 && (97 <= Int(lowercase(check)[1]) <= 122)\nend\n\nfunction add_one(i)    check = split(txt, " ")[end]\n    return length(check) == 1 && (97 <= Int(lowercase(check)[1]) <= 122)\nend"
    # a = t.extract_functions(s)

    # s = "    function longest(strings::Vector{String})::Union{String, Nothing}\n        if isempty(strings)\n            return nothing\n        end\n    \n        maxlen = maximum(length.(strings))\n        for s in strings\n            if length(s) == maxlen\n                return s\n            end\n        end\n    \n        return nothing\n    end\n"
    # a = t.completion_touchup(s, "")

    # s="\n\n### Julia version\n\nfunction solution(lst::Vector{Int64})::Int64\n    # filter the list\n    filtered_lst = filter((idx, x) -> idx % 2 == 0 && x % 2 == 1, enumerate(lst))\n    \n    # sum the filtered list\n    sum_lst = sum(x[2] for x in filtered_lst)\n    \n    # return the sum\n    return sum_lst\nend\n\nNote that the function signature remains the same as the original Julia code. The steps I provided were used to filter the list, sum the filtered list, and return the sum as the output of the function.\n"
    # a = t.completion_touchup(s, "")

    # s ="\nfunction encode(message::String)::String\n    vowels = \"aeiouAEIOU\"\n    vowels_replace = Dict([(i, Char(UInt16(i) + 2)) for i in vowels])\n    message = lowercase(message) |> swapcase\n    encoded_message = [haskey(vowels_replace, i) ? vowels_replace[i] : i for i in message]\n    return join(encoded_message)\nend\n"
    # a = t.completion_touchup(s, "")
    pass