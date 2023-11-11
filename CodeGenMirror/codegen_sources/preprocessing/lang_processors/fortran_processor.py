# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pygments import highlight, lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor
from codegen_sources.preprocessing.obfuscation.bobskater_obfuscator import (
    obfuscateString,
)
from codegen_sources.preprocessing.obfuscation.utils_deobfuscation import dico_to_string

import tokenize
from io import BytesIO
import re


class FortranProcessor(LangProcessor):
    def __init__(self, root_folder=None):
        self.lexer = get_lexer_by_name("fortran")
        self.language = "fortran"
        # other fortran libraries:
        # - parser: https://fparser.readthedocs.io/en/latest/fparser2.html

    @staticmethod
    def process_tokens(tokens):
        if not isinstance(tokens, list):
            tokens = list(tokens
                          )
        def is_operator(i, tup, tokens):
            return (tup[0] == Token.Punctuation and tup[1] == ".") and \
               (tokens[i-2][0] == Token.Punctuation and tokens[i-2][1] == ".") and \
               (tokens[i - 1][0] == Token.Name.Builtin)

        out = []
        for i, tup in enumerate(tokens):
            if is_operator(i, tup, tokens):
                out = out[:-2]
                tup = (tup[0], f".{tokens[i-1][1]}.")
            if tup[0] == Token.Text.Whitespace and tup[1] == " ":
                continue
            else:
                surface_form = tup[1].strip()
                if len(surface_form) > 0:
                    out.append(surface_form)
                if tup[1] != surface_form:
                    left = tup[1].replace(surface_form, "")
                    left = left.replace("\n","NEWLINE,").replace("\t","INDENT,").replace("    ","INDENT,").strip()
                    if len(left) > 0:
                        out.extend(left[:-1].split(","))

        return out

    def tokenize_code(self, code, keep_comments=False, process_strings=True, tokens=None):
        assert isinstance(code, str)
        if tokens is None:
            try:
                tokens = lex(code, self.lexer)
                tokens = self.process_tokens(tokens)
            except:  # for some reason above code does not work with multi-process, need to initiate lexer here
                tokens = lex(code, get_lexer_by_name("fortran"))
                tokens = self.process_tokens(tokens)
        else:
            tokens = self.process_tokens(tokens)
        return tokens

    def detokenize_code(self, code):
        # fortran doesn't care about indentation but cares about new line,
        # but for readability we are going to follow python tokenization rules
        assert isinstance(code, str) or isinstance(code, list)
        if isinstance(code, list):
            code = " ".join(code)

        code = code.replace("NEWLINE", "\n").replace("INDENT", "    ")
        return code

    def obfuscate_code(self, code):
        raise NotImplementedError

    def extract_functions(self, tokenized_code: str):
        """Extract functions from tokenized Cobol code"""
        raise NotImplementedError

    def get_function_name(self, function):
        assert isinstance(function, str) or isinstance(function, list)
        if isinstance(function, list):
            function = " ".join(function)
        return function.split("(")[0].split()[-1].strip()

    def extract_arguments(self, function):
        raise NotImplementedError


# if __name__ == "__main__":
#     func = """subroutine f_fortran(multiples, sum)
#     integer, intent(in) :: multiples(:)
#     integer, intent(out) :: sum
#     sum = 0
#     do i = 1, size(multiples)
#         sum = sum + multiples(i)
#     end do
# end subroutine f_fortran"""
#     func = """function f_fortran(str) result(res)
#     character(len=*), intent(in) :: str
#     integer :: i, zeros, ones
#     character(len=1) :: ch
#     logical :: res
#
#     zeros = 0
#     ones = 0
#     do i = 1, len(str)
#         ch = str(i:i)
#         if (ch == '0') then
#             zeros = zeros + 1
#         else
#             ones = ones + 1
#         end if
#     end do
#     res = (zeros == 1 .or. ones == 1)
# end function f_fortran"""
#     p = FortranProcessor()
#     tokens = " ".join(p.tokenize_code(func))
#     code = p.detokenize_code(tokens)
#     pass