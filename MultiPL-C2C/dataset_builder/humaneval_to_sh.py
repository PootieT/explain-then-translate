# This script translates problems from the OpenAI HumanEval dataset into bash.
#
# Installed version is:
# GNU bash, version 5.1.16(1)-release (x86_64-pc-linux-gnu)
#
# Note: We evaluate with Bash 4.2
#
# Bash is not a general-purpose programming language, and has its own quirks to work around.
# In particular, the main datatype is string. While Bash has arrays and associative arrays,
# the convention is to use strings with spaces and/or newlines as separators.
#
# This makes the translation tricky, as we can't pass strings around and concatenate them.
# If we have a list, we need to know if its elements are lists or base values, since
# that determines the list encoding. Therefore, we defer translation and pass Python
# values around. Only deep_equality() and gen_call() need to return strings, so
# they call py_to_bash() to perform the translation.
#
# Known limitations:
#   1. List elements containing whitespace are not supported
#   2. More than two layers of nesting is not supported
#   3. Nested dictionaries are not supported
#
# 2 and 3 do not occur in the dataset, but 1 does.
# The translator tries to check for these cases and terminates if they are encountered.
import pdb
import re
import ast
from typing import List

# We turn multi-line docstrings into single-line comments. This captures the
# start of the line.
from pygments import lex
from pygments.lexers import get_lexer_by_name

from dataset_builder.utils import rebalance_brackets, remove_pre_function_comments, remove_trailing_comments, \
    remove_tests

DOCSTRING_LINESTART_RE = re.compile("""\n(\s+)""")

WHITESPACE = [' ', '\t', '\n', '\r']

def contains_whitespace(s):
    return any(ws in s for ws in WHITESPACE)

def is_quoted(s):
    return len(s) > 0 and s[0] == '"' and s[-1] == '"'

def quote(s):
    return s if is_quoted(s) else '"' + s + '"'

def unquote(s):
    return s[1:-1] if is_quoted(s) else s

# Bash does not have type annotations, but we include comments explaining the encoding the tests use.
# List and Tuple -> space-separated list
# Nested list -> newline-separated, space-separated list
# Dictionary -> two column CSV in key,value order
def type_to_comment(t, i):
    match t:
        case ast.Subscript(ast.Name(id), slice, ctx):
            match id:
                case "List":
                    match slice:
                        case ast.Subscript(ast.Name(id2), slice2, ctx2):
                            return f"# ${i} is a newline-separated, space-separated list\n"
                    return f"# ${i} is a space-separated list\n"
                case "Tuple":
                    return f"# ${i} is a space-separated list\n"
                case "Dict":
                    return f"# ${i} is a two column CSV in key,value order\n"
                case other:
                    return f"# ${i} is an argument\n"
        case ast.Name("int"):
            return f"# ${i} is an integer\n"
        case ast.Name("float"):
            return f"# ${i} is a floating point\n"
        case ast.Name("str"):
            return f"# ${i} is a string\n"
        case ast.Name(x):
            return f"# ${i} is a ${x}\n"
        case other:
            return f"# ${i} is an argument\n"

# Translation of Python values into Bash strings is deferred until this point.
# Non-list values have already been translated to Bash syntax.
# There are three cases:
#   1.  Empty list: this is translated to an empty string.
#   2a. List of dictionaries: not supported
#   2b. Nested list with more than two layers of nesting: not supported
#   2c. Nested list: the inner lists are recursively translated (which results in
#       a quoted, space-separated string), unquoted, and then joined with newlines.
#   2d. List: the elements are recursively translated (which results in quoted strings),
#       unquoted, and then joined with spaces.
#   3a. Nested dictionary: not supported
#   3b. Dictionary: the dictionary is translated to CSV format, in key,value order.
def py_to_bash(val):
    if type(val) is list and len(val) == 0:
        return '""'
    elif type(val) is list:
        if all(type(v) is dict for v in val):
            raise Exception("Cannot translate list with dictionary as elements")
        elif all(type(vv) is list for v in val for vv in v):
            raise Exception("Cannot translate list with more than two layers of nesting")
        elif all(type(v) is list for v in val):
            # Translate list to a string where the inner list is space separated and the outer list is newline separated
            return quote("\\n".join([unquote(py_to_bash(v)) for v in val]))
        else:
            # Translate list to a string with elements separated by spaces
            return quote(" ".join([unquote(py_to_bash(v)) for v in val]))
    elif type(val) is dict:
        if any(type(v) is not str for v in val.values()):
            raise Exception("Cannot translate nested dictionary")
        # Translate dictionary to CSV format
        return quote("\\n".join(unquote(k) + "," + unquote(v) for k, v in val.items()))
    else:
        return val

class Translator:

    stop = [
        # completion stop tokens
        # "\n}"  # does not allow helper function generation, but
        "\n#",  # added, alternative that allows helper function
        "\nAnswer",
        "\necho"
        "\n```"
        # chat stop tokens
        # "###",
        # "\n# example usage",  # starts generating unit tests
        # "\n# Example usage",
        # "\n# Call the"
    ]

    def __init__(self):
        self.num = 0
        self.entry_point = ""
        self.skip_annotations = False
        self._lexer = None

    @property
    def lexer(self):
        if self._lexer is None:
            self._lexer = get_lexer_by_name("bash")
        return self._lexer

    def tokenize_code(self, script):
        return [t[1].strip() for t in lex(script, self.lexer)]

    def file_ext(self):
        return "sh"

    def translate_prompt(self, name: str, args: List[ast.arg], _returns, description: str) -> str:
        bash_description = (
            "#!/bin/bash\n# " + re.sub(DOCSTRING_LINESTART_RE, "\n# ", description.strip()) + "\n"
        )
        if self.skip_annotations == False:
            annotations = [type_to_comment(arg.annotation, i + 1) for i, arg in enumerate(args)]
            annotations = "#\n" + "".join(annotations) if len(annotations) > 0 else ""
        else:
            annotations = ""
        return f"{bash_description}{annotations}{name}() {{\n"

    def test_suite_prefix_lines(self, entry_point) -> List[str]:
        """
        This code goes at the start of the test suite.

        We define candidate as a wrapper function, that forwards its arguments to entry_point.
        Tests are written in the run_test function, and we use "set -e" to halt execution and
        return a non-zero status if anything fails.
        """
        return [
            # Need a closing brace, because we used it as a stop token
            # "}",  # if completion, use this, if chat completion, comment this out
            "",
            "candidate() {",
            f"    {entry_point} \"$@\"",
            "}",
            "",
            "set -e",
            "run_test() {",
        ]

    def test_suite_suffix_lines(self) -> List[str]:
        return [
            "}",
            "",
            "run_test"
        ]

    def deep_equality(self, left: str, right: str) -> str:
        """
        All tests are assertions that compare deep equality between left and right.

        Bash is tricky, because there is no deep equality, so we just compare strings.

        Tests are of the form:
            [[ left = right ]]

        Up until this point, we've been passing Python values around.
        Now we need to do the actual translation. This allows us to keep track of the list structure.
        """
        return "    [[ " + py_to_bash(left) + " = " + py_to_bash(right) + " ]]"

    def gen_literal(self, c: bool | str | int | float) -> str:
        """Translate a literal expression
        c: is the literal value
        """
        res = repr(c)
        if type(c) == bool:
            res = str(c).lower()
        elif type(c) == str:
            # Escape strings
            res = c.replace('\\', '\\\\').replace('\n', '\\n').replace('"', '\\"').replace('$', '\\$').replace('!', '\\!')
        elif c is None:
            res = "None"
        # Quote everything
        return quote(res)

    def gen_var(self, v: str) -> str:
        """Translate a variable with name v."""
        return v

    def gen_list(self, l: List):
        """Translate a list with elements l
        Ultimately, a list [x, y, z] translates to a space-separated list "x y z"
        Ultimately, a nested list [[a, b], [c, d]] translates to a newline-separated, space-separated list "a b\nc d"

        Because we need to know what values we're passing around, pass around Python values.

        If any element contains whitespace, then we can't translate it.
        """
        if any(contains_whitespace(ll) for ll in l):
            raise Exception("Cannot translate list element that contains whitespace")
        return l

    def gen_tuple(self, t: List):
        """Translate a tuple with elements t
        A tuple (x, y, z) translates to a space-separated list "x y z"

        Because we need to know what values we're passing around, pass around Python values.

        If any element contains whitespace, then we can't translate it.
        """
        if any(contains_whitespace(tt) for tt in t):
            raise Exception("Cannot translate tuple element that contains whitespace")
        return t

    def gen_dict(self, keys: List[str], values: List[str]) -> str:
        """Translate a dictionary with keys and values
        A dictionary { "key1": val1, "key2": val2 } translates to a CSV: key1,val1\nkey2,val2

        Because we need to know what values we're passing around, pass around Python values.

        If any element contains whitespace, then we can't translate it.
        """
        if any(contains_whitespace(k) or contains_whitespace(v) for k, v in zip(keys, values)):
            raise Exception("Cannot translate list element that contains whitespace")
        return dict(zip(keys, values))

    def gen_call(self, func: str, args: List) -> str:
        """Translate a function call `func(args)`
        A function call f(x, y, z) translates to $(f x y z)

        Up until this point, we've been passing Python values around.
        Now we need to do the actual translation. This allows us to keep track of the list structure.
        """
        return "$(" + func + " " + " ".join([py_to_bash(a) for a in args]) + ")"

    def no_completion_prompt_stub(self):
        return "echo 0"

    def get_function_name(self, line: str):
        assert self.is_function_signature(line)
        return line.split("(")[0].split()[-1].strip()

    def is_function_signature(self, line: str):
        line = line.strip()
        return line.endswith("() {") or line.endswith("(){")

    def is_end_of_function(self, line: str):
        return line.startswith("}")

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
                if "{" in tokens or "${#" in tokens or "${" in tokens:
                    bracket_stack += 1
            if curr_func is not None and bracket_stack == 0:
                # function completed!
                functions.append("\n".join(curr_func))
                curr_func = None
        return functions

    @staticmethod
    def remove_imports(script: str):
        return script

    @staticmethod
    def remove_class_decl(script):
        return script

    @staticmethod
    def is_comment(line: str):
        return line.strip().startswith("#")

    def completion_touchup(self, script:str, prompt: str):
        script = script.strip()

        # remove preceeding comments to before class/function definition
        script = remove_pre_function_comments(script, self.is_function_signature, self.is_comment)

        # calculate the amount of left and right brackets, compared to expectation, and add or delete right
        # brackets if needed
        tokens = self.tokenize_code(script)
        left_count, right_count = tokens.count("{")+tokens.count("${#")+tokens.count("${"), tokens.count("}")
        if prompt.strip().split("\n")[-1] in script:
            expected_diff = 0  # output should have one more left than right bracket
        else:
            expected_diff = 1

        add_cnt = (left_count - right_count) + expected_diff
        script = rebalance_brackets(script, add_cnt)

        script = remove_trailing_comments(script)
        return script

    @staticmethod
    def remove_tests(script:str):
        return remove_tests(script, "candidate() {", "")


if __name__ == "__main__":
    # t = Translator()
    # x = '#!/bin/bash\n# \n#\n# $1 is a space-separated list\nnumerical_letter_grade() {\n    # Create an empty array to store the letter grades\n    letter_grade=()\n\n    # Loop through each grade in the input list\n    for gpa in $1; do\n        # Use if-elif statements to determine the corresponding letter grade based on the GPA\n        if (( $(echo "$gpa == 4.0" | bc -l) )); then\n            letter_grade+=("A+")\n        elif (( $(echo "$gpa > 3.7" | bc -l) )); then\n            letter_grade+=("A")\n        elif (( $(echo "$gpa > 3.3" | bc -l) )); then\n            letter_grade+=("A-")\n        elif (( $(echo "$gpa > 3.0" | bc -l) )); then\n            letter_grade+=("B+")\n        elif (( $(echo "$gpa > 2.7" | bc -l) )); then\n            letter_grade+=("B")\n        elif (( $(echo "$gpa > 2.3" | bc -l) )); then\n            letter_grade+=("B-")\n        elif (( $(echo "$gpa > 2.0" | bc -l) )); then\n            letter_grade+=("C+")\n        elif (( $(echo "$gpa > 1.7" | bc -l) )); then\n            letter_grade+=("C")\n        elif (( $(echo "$gpa > 1.3" | bc -l) )); then\n            letter_grade+=("C-")\n        elif (( $(echo "$gpa > 1.0" | bc -l) )); then\n            letter_grade+=("D+")\n        elif (( $(echo "$gpa > 0.7" | bc -l) )); then\n            letter_grade+=("D")\n        elif (( $(echo "$gpa > 0.0" | bc -l) )); then\n            letter_grade+=("D-")\n        else\n            letter_grade+=("E")\n        fi\n    done\n\n    # Return the list of letter grades\n    echo "${letter_grade[@]}"\n}\n\n# Call the function with the input list as the argument\nnumerical_letter_grade "$1"\n\nNote that I used the `bc` command to perform floating-point comparisons in the if-elif statements.\n'
    # a = t.remove_trailing_comments(x)
    pass
