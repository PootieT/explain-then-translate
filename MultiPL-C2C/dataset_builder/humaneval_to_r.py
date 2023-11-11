# This script translates problems from the OpenAI HumanEval dataset into R.
import ast
import re
from typing import List

from pygments import lex
from pygments.lexers import get_lexer_by_name

from dataset_builder.utils import remove_pre_function_comments, rebalance_brackets, remove_trailing_comments, \
    remove_tests

DOCSTRING_LINESTART_RE = re.compile("""\n(\s+)""")

# needs_hashmap = False
# basic = ["number","string","boolean"]
#
# def translate_type(t):
#     global needs_hashmap
#     match t:
#         case ast.Subscript(ast.Name(id), slice, ctx):
#             match id:
#                 case "List":
#                     if translate_type(slice) not in basic:
#                         return "complex"
#                     return "list"
#                 case "Union":
#                     return "union"
#                 case "Tuple":
#                     return "tuple"
#                 case "Dict":
#                     return "dict"
#                 case "Optional":
#                     return "undefined"
#                 case other:
#                     raise Exception(f"Bad generic {other}")
#         case ast.Name("int") | "int":
#             return "number"
#         case ast.Name("float"):
#             return "number"
#         case ast.Name("bool"):
#             return "boolean"
#         case ast.Name("str") | "str":
#             return "string"
#         case None:
#             return "no type"
#         case ast.Name("Any"):
#             return "any"
#         case ast.Name(x):
#             raise Exception(f"unknown name {x}")
#         case ast.Constant(Ellipsis):
#             return "ellipsis"
#         case _other:
#             raise Exception(f"unknown annotation: {t}")

def coerce(expr: str, type) -> str:
    match expr, type:
        case expr, ast.Subscript(ast.Name("List"), sub):
            if "id" not in sub._fields:
                #print(expr,sub)
                return expr
            elif "complex" in sub.id or "Any" in sub.id:
                return expr
            #print("FOUND VECTOR",expr)
            return coerce_to_vector(expr)
        case _:
            return expr
def coerce_to_vector(expr):
    return 'c('+'('.join(expr.split('(')[1:]) 

class Translator:
    '''R Translator
    '''

    stop = [
        # regular completion stops
        # '\n#',
        # '\n```',
        # '\nTry',  # added
        '\n}',  # added, since R allows nested function, hard stop with end of function make sense
        # chat completion stops
        # "### steps",
        # "### Steps"
    ]

    def file_ext(self):
        return "r"

    def __init__(self):
        # global needs_hashmap
        self.type = None
        self._lexer = None

    @property
    def lexer(self):
        if self._lexer is None:
            self._lexer = get_lexer_by_name("r")
        return self._lexer

    def tokenize_code(self, script):
        return [t[1].strip() for t in lex(script, self.lexer)]

    @staticmethod
    def module_imports() -> str:
        return "\n".join([
            "library(sets)",
            "library(stringi)",
            "suppressPackageStartupMessages(library(R.utils))",  # suppressed a lot of things
            "suppressPackageStartupMessages(library(stringr))",   # suppresses %>% function, not used in most functions i think
            "suppressPackageStartupMessages(library(hash))",
        ]) + "\n"

    def translate_prompt(self, name: str, args: List[ast.arg], returns, description: str) -> str:
        # global needs_hashmap
        r_description = (
            "# " + re.sub(DOCSTRING_LINESTART_RE, "\n# ", description.strip()) + "\n"
        )
        if len(description) == 0:
            r_description = ""
        # needs_hashmap = False
        self.type = [[arg.annotation for arg in args], returns]
        
        arg_names = [arg.arg for arg in args]
        arg_list = ", ".join(arg_names)
        return f"{self.module_imports()}{r_description}{name} <- function({arg_list})" + ' {\n'
        # return f"{r_description}{name} <- function({arg_list})" + ' {\n'

    def test_suite_prefix_lines(self, entry_point) -> List[str]:
        """
        This code goes at the start of the test suite.
        """
        return [
            "",
            "}",
            "test_humaneval <- function() {",
            f"    candidate <- {entry_point}",
        ]

    def test_suite_suffix_lines(self) -> List[str]:
        return ["}", "test_humaneval()"]

    def deep_equality(self, left: str, right: str) -> str:
        """
        All tests are assertions that compare deep equality between left and right.
        if right is an numerical variable (Integer or Double), use == instead of identical.
        Make sure you use the right equality operator for your language. For example,
        == is the wrong operator for Java and OCaml.
        """
        # if not any([right.startswith(w) for w in ["c(", "list(", "NULL"]]):
        #     return "    if(!({} == {}))".format(left, right) + "{quit('no', 1)}"
        # return "    if(!identical({}, {}))".format(left, right) + "{quit('no', 1)}"
        return "    stopifnot(isTRUE(all.equal({}, {})))".format(left, right)

    def gen_literal(self, c):
        ''' Translate a literal expression
            c: is the literal value
        '''
        if type(c) == bool:
            return 'TRUE' if c else 'FALSE'
        elif c is None:
            return 'NULL'
        return repr(c)
    
    def gen_var(self, v):
        '''Translate a variable with name v.
        '''
        return v

    def _get_r_type(self, e:str):
        if e.startswith("c("):
            return "vector"
        elif e.startswith("list("):
            return "list"
        elif e == "NULL":
            return "null"
        elif e == "TRUE" or e == "FALSE":
            return "boolean"
        else:
            # https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-represents-a-number-float-or-int
            return "numeric" if e.replace("-","",1).replace('.','',1).isdigit() else "string"

    def is_atomic(self, l):
        '''inputs are all strings, but we need to determine what type they are in R
        '''
        type_set = set([self._get_r_type(e) for e in l])
        if "null" in type_set or "vector" in type_set or "list" in type_set:
            return False
        return len(type_set) <= 1

    def gen_list(self, l):
        '''Translate a list with elements l
        A list [ x, y, z ] translates to:
        - c(x, y, z) if x, y, and z have the same type
        - list(x, y, z) otherwise
        '''
        if self.is_atomic(l):
           return "c(" + ", ".join(l) + ")"
        return "list(" + ", ".join(l) + ")"
   
    #there are no r tuples, but r lists are mostly immutable?
    def gen_tuple(self, t):
        '''Translate a tuple with elements t
           A tuple (x, y, z) translates to:
        - c(x, y, z) if x, y, and z have the same type
        - list(x, y, z) otherwise
        '''
        if self.is_atomic(t):
           return "c(" + ", ".join(t) + ")"
        return "list(" + ", ".join(t) + ")"
    
    def gen_dict(self, keys, values):
        '''Translate a dictionary with keys and values (uses R list with keys)
           A dictionary { "key1": val1, "key2": val2 } translates to list("key1" = val1, "key2" = val2)  
        '''
        return "list(" + ", ".join(f'{k} = {v}' for k, v in zip(keys, values)) + ")"
    
    def gen_call(self, func, args):
        '''Translate a function call `func(args)`
           A function call f(x, y, z) translates to f(x, y, z)
        '''
        args = [coerce(a,self.type[0][i]) for i,a in enumerate(args)]
        return func + "(" + ", ".join(args) + ")"

    def get_function_name(self, line: str):
        assert self.is_function_signature(line)
        return line.split("<-")[0].split()[-1].strip()

    def is_function_signature(self, line: str):
        # failure case: if { is on the next line
        line = line.strip()
        return (line.endswith(") {") or line.endswith("){")) and ("function" in line) and ("<-" in line)

    def extract_functions(self, script: str) -> List[str]:
        # copied from java
        functions = []
        curr_func = None
        bracket_stack = 0
        for l in script.split("\n"):
            if self.is_function_signature(l):
                if bracket_stack == 0:
                    curr_func = [l]
                else:
                    # java does not allow nested functions, but could be generated. we just treat
                    # this as part of the parent function.
                    curr_func.append(l)
            else:
                if curr_func is not None:
                    curr_func.append(l)
            if curr_func is not None:
                # if we are in a function, keep track of {}s to know when it ends
                tokens = set(self.tokenize_code(l))
                if "}" in tokens:
                    bracket_stack -= 1
                if "{" in tokens:
                    bracket_stack += 1
            if curr_func is not None and bracket_stack == 0:
                # function completed!
                functions.append("\n".join(curr_func))
                curr_func = None
        return functions

    @staticmethod
    def remove_imports(script: str):
        return "\n".join([l for l in script.split("\n") if not (l.strip().startswith("library") or l.strip().startswith("import"))])

    @staticmethod
    def remove_class_decl(script):
        return script

    @staticmethod
    def is_comment(line: str):
        return line.strip().startswith("#")

    def completion_touchup(self, script: str, prompt: str):
        script = script.strip("\n")

        # remove preceeding comments to before class/function definition
        script = remove_pre_function_comments(script, self.is_function_signature, self.is_comment)

        # calculate the amount of left and right brackets, compared to expectation, and add or delete right
        # brackets if needed
        tokens = self.tokenize_code(script)
        left_count, right_count = tokens.count("{"), tokens.count("}")
        if prompt.strip().split("\n")[-1] in script:
            expected_diff = 0  # output should have one more left than right bracket
        else:
            expected_diff = 1

        add_cnt = (left_count - right_count) + expected_diff
        script = rebalance_brackets(script, add_cnt)

        script = remove_trailing_comments(script)
        return script

    @staticmethod
    def remove_tests(script: str):
        return remove_tests(script, "test_humaneval <- function() {", "")
