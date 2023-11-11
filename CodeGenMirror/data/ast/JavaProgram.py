from typing import List

from codegen_sources.model.src.utils import TREE_SITTER_ROOT
from codegen_sources.preprocessing.lang_processors.java_processor import JavaProcessor
from data.ast.Program import Program
import copy

JAVA_CONDITION_KEYWORDS = {"while", "if", "else", "for"}

class JavaProgram(Program):
    processor = JavaProcessor(TREE_SITTER_ROOT)

    def __init__(self, input_program, obfuscate = False, bpe_model = None):
        if type(input_program) == List:
            input_program = " ".join(input_program)
        input_program = input_program.replace("@@ ", "")
        if "public class" not in input_program:
            input_program = "public class Main {\n" + input_program + "\n}"
            input_program = self.detokenize_code(input_program)
        super().__init__(input_program=input_program, obfuscate=obfuscate, bpe_model=bpe_model)

    @staticmethod
    def parse_program_stmt(prog: List[str]) -> List[List[str]]:
        """
        Parses a program into a list of statements (each is a list of tokens).
        Note the individual tokens should not be BPE tokens, they should be lexer tokens.
        """
        remaining = copy.deepcopy(prog)
        indiv_lines = []
        next_line = []
        ctr = len(remaining)
        in_condition = False
        open_condition = False
        in_array_literal = False
        condition_depth = 0
        array_depth = 0
        in_quote = False
        quotechar = None
        while remaining:
            next_token = remaining.pop(0)

            if next_token in ("\"", "'"):
                if not in_quote:
                    in_quote = True
                    quotechar = next_token
                else:
                    if quotechar == next_token:
                        in_quote = False
                        quotechar = None
                next_line.append(next_token)

            elif in_quote:
                next_line.append(next_token)

            elif next_token in JAVA_CONDITION_KEYWORDS:
                in_condition = True
                next_line.append(next_token)

            elif in_condition and next_token == "(":
                open_condition = True
                condition_depth += 1
                next_line.append(next_token)

            elif in_condition and open_condition and next_token == ")":
                next_line.append(next_token)
                if remaining[0] == "{":
                    next_line.append(remaining.pop(0))
                condition_depth -= 1
                if condition_depth == 0:
                    indiv_lines.append(next_line)
                    next_line = []
                    in_condition = False
                    open_condition = False
                    in_array_literal = False

            elif next_token == "[":
                next_line.append(next_token)
                if remaining[0] == "]":
                    next_line.append(remaining.pop(0))
                for e in remaining:
                    # end of stmt w/out assignment -> not array literal
                    if e == ";":
                        break
                    elif e == "=":
                        in_array_literal = True
                        break
                    else:
                        pass

            elif in_array_literal and next_token == "{":
                array_depth += 1

            elif in_array_literal and next_token == "}":
                next_line.append(next_token)
                array_depth -= 1
                if array_depth == 0:
                    in_array_literal = False

            elif next_token == "}":
                assert next_line == [], f"next token was }}, next_line is {next_line}, remaining dump is {remaining}, and indiv_lines dump is {indiv_lines} in_array_literal is {in_array_literal}"
                indiv_lines.append([next_token])

            elif ((next_token == ";" and not open_condition)
                  or (next_token == "{" and not in_array_literal)
                  or (next_token == ":" and ("case" in next_line or "default" in next_line))):
                next_line.append(next_token)
                indiv_lines.append(next_line)
                next_line = []
                in_array_literal = False
            else:
                next_line.append(next_token)

            ctr -= 1
            if ctr < 0:
                raise RuntimeError(f"ctr went below 0, next token is {next_token}, next line was {next_line}, remaining is {remaining}, and indiv_lines dump is {indiv_lines}")
        return indiv_lines





# def __init__(self, input_program = None, obfuscate=False, bpe_model=None):
    #     super().__init__(input_program, obfuscate, bpe_model)


