from codegen_sources.model.src.utils import TREE_SITTER_ROOT
from codegen_sources.preprocessing.lang_processors.python_processor import PythonProcessor
from data.ast.Program import Program
import copy
from typing import List

PYTHON_TOKENS = ["DEDENT", "INDENT", "NEWLINE"]

class PythonProgram(Program):
    processor = PythonProcessor(TREE_SITTER_ROOT)

    def __init__(self, input_program, obfuscate = False, bpe_model = None):
        self.input_program = input_program
        if type(self.input_program) == list:
            self.input_program = " ".join(self.input_program)
        self.input_program.replace("@@ ", "")
        if any([token in input_program for token in PYTHON_TOKENS]):
            input_program = self.processor.detokenize_code(input_program)
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
        while remaining:
            next_token = remaining.pop(0)
            if next_token == "DEDENT":
                assert next_line == [], f"next token was DEDENT, next_line is {next_line}, remaining dump is {remaining}, and indiv_lines dump is {indiv_lines}"
                indiv_lines.append([next_token])
            elif next_token == ":" and remaining[0] == "NEW_LINE":
                assert remaining[
                           1] == "INDENT", f"next token was a colon ':' followed by NEW_LINE, but didn't indent after: {remaining[1]}, remaining was {remaining} and indiv_lines dump is {indiv_lines}"
                next_line.append(next_token)
                next_line.append(remaining.pop(0))
                next_line.append(remaining.pop(0))
                indiv_lines.append(next_line)
                next_line = []
            elif next_token == "NEW_LINE" or next_token == "ENDCOM":
                next_line.append(next_token)
                if remaining and remaining[0] == "NEW_LINE":
                    next_line.append(remaining.pop(0))
                indiv_lines.append(next_line)
                next_line = []
            else:
                next_line.append(next_token)

            ctr -= 1
            if ctr < 0:
                raise RuntimeError(
                    f"ctr went below 0, next token is {next_token}, next line was {next_line}, remaining is {remaining}, and indiv_lines dump is {indiv_lines}")
        assert next_line == [], f"finished but next line was not empty: next_line is {next_line}, remaining dump is {remaining}, and indiv_lines dump is {indiv_lines}"
        return indiv_lines


