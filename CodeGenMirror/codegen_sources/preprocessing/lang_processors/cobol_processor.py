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


class CobolProcessor(LangProcessor):
    def __init__(self, root_folder=None):
        self.lexer = get_lexer_by_name("cobolfree")
        self.language = "cobol"

    def tokenize_code(self, code, keep_comments=False, process_strings=True, tokens=None):
        assert isinstance(code, str)
        # TODO tokenized cobol could include indent/ dedent to improve readability
        keep_space = False
        keep_newline = False
        remove_token = lambda tup: (not keep_space and tup[0] == Token.Text.Whitespace) or (
                not keep_newline and tup[1] == "\n"
        )
        if tokens is None:
            try:
                tokens = lex(code, self.lexer)
                tokens = [t[1].strip() for t in tokens if not remove_token(t)]
            except:  # for some reason above code does not work with multi-process, need to initiate lexer here
                tokens = lex(code, get_lexer_by_name("cobolfree"))
                tokens = [t[1].strip() for t in tokens if not remove_token(t)]
        else:
            tokens = [t[1].strip() for t in tokens if not remove_token(t)]
        return tokens

    def detokenize_code(self, code):
        # replace recreate lines with \n and appropriate indent / dedent
        # removing indent/ dedent tokens
        assert isinstance(code, str) or isinstance(code, list)
        if isinstance(code, list):
            code = " ".join(code)
        # Cobol doens't care about tabs / space / new line, but if need readability,
        # may need other solutions
        return code

    def obfuscate_code(self, code):
        # disclaimer, this is a very roughly written obfuscator that only deals with single function
        # only replaces function name/ call with FUNC_X and variable with VAR_X. and it doesn't consider
        # any executional level syntax (i.e. a = 1, if a is seen before it will be replaced as an old
        # variable as opposed to new variable (as it should be))
        # TODO currently it returns only the tokenized code, in future maybe improve it to returning original code
        tokens = list(lex(code, self.lexer))
        vars_dico = {}
        funcs_dico = {}
        for i, t in enumerate(tokens):
            if t[0] in [Token.Name.Variable, Token.Name.Function]:  # sometimes though variable_names shadow some built-in function
                if tokens[i-3][1].strip() == "PROGRAM-ID" and t[1].strip() not in funcs_dico:
                    funcs_dico[t[1].strip()] = f"FUNC_{len(funcs_dico)}"
                elif t[1].strip() not in vars_dico:  # variable
                    vars_dico[t[1].strip()] = f"VAR_{len(vars_dico)}"
            elif t[0] == Token.Literal.String.Double and tokens[i-1][1].strip() == "CALL":  # call function
                surface_form = t[1].strip().replace("'", "").replace('"',"")
                if surface_form in funcs_dico:
                    tokens[i] = (t[0], '"' + funcs_dico[surface_form] + '"')

            if t[1].strip() in vars_dico:
                tokens[i] = (t[0], vars_dico[t[1].strip()])
            if t[1].strip() in funcs_dico:
                tokens[i] = (t[0], funcs_dico[t[1].strip()])

        obf_code = " ".join(self.tokenize_code("", tokens=tokens))
        vars_dico.update(funcs_dico)
        dico = " | ".join([f"{v} {k}" for k, v in vars_dico.items()])
        return obf_code, dico

    def extract_functions(self, tokenized_code: str):
        """Extract functions from tokenized Cobol code"""
        raise NotImplementedError

    def get_function_name(self, function):
        assert isinstance(function, str) or isinstance(function, list)
        if isinstance(function, str):
            function = function.split()
        func_name = function[function.index("PROGRAM-ID") + 2]

        # really it's a bug if the input function is not tokenized, if input is tokenized
        # there shouldn't be this issue
        # func_name = func_name.split("DATA DIVISION")[0].split("PROGRAM-ID.").replace(".", "")
        return func_name

    def extract_arguments(self, function):
        tokens = list(lex(function, self.lexer))
        start = tokens.index((Token.Keyword.Reserved, 'LINKAGE')) + 4
        end = tokens.index((Token.Keyword.Reserved, 'PROCEDURE'))
        arg_tokens = tokens[start:end]
        arg_types = []
        arg_names = []
        is_table = False
        for i, (token_type, text) in enumerate(arg_tokens):
            # error and name.function are either fault of lexer or fault of program writtern, but we keep them
            if token_type in (Token.Name.Variable, Token.Error, Token.Name.Function) and text!="X":
                if text.endswith("_table"):
                    is_table = True
                    continue
                if "88" == arg_tokens[i-1][1].strip():  # skip booleans
                    continue

                var_name = text.strip()
                var_type = arg_tokens[i+2][1].strip()
                if var_type == "USAGE":  # if it's USAGE COMP-1 or COMP-2
                    var_type = arg_tokens[i + 4][1].strip()
                elif var_type == "PIC X" and  f"88 {var_name}_" in function:
                    var_type = "boolean"
                if is_table:
                    var_type = "table " + var_type
                    is_table = False
                elif i+5 < len(arg_tokens) and arg_tokens[i+5][1].strip() == "88" and "boolean" not in var_type:
                    var_type = "boolean " + var_type
                if var_type == "USAGE":
                    var_type = arg_tokens[i+4][1].strip()
                arg_types.append(var_type)
                arg_names.append(var_name)

        return [arg_types, arg_names]


if __name__ == "__main__":
#     func = """
# IDENTIFICATION DIVISION.
# PROGRAM-ID. swap.
# DATA DIVISION.
# WORKING-STORAGE SECTION.
#     01 tmp PIC S9(4) COMP.
#
# LINKAGE SECTION.
# 	01 array_table.
# 	    02 array PIC S9(4) COMP OCCURS 100.
# 	01 i PIC S9(9) COMP.
# 	01 j PIC S9(9) COMP.
#
# PROCEDURE DIVISION USING array_table, i, j.
# 	MOVE array(i) TO tmp.
# 	MOVE array(j) TO array(i).
# 	MOVE tmp TO array(j).
# 	GOBACK.
# """
#     func = """
# IDENTIFICATION DIVISION.
# PROGRAM-ID. CLASS_023fa3df801cfbc2fb6-TEST.
#
# ENVIRONMENT DIVISION.
#
# DATA DIVISION.
# LINKAGE SECTION.
#     01 loopIdx PIC S9(9).
#     01 boolean1 PIC X.
#         88 boolean1_false VALUE X'00'.
#         88 boolean1_true VALUE X'01' THROUGH X'FF'.
#
#     01 boolean0 PIC X.
#         88 boolean0_false VALUE X'00'.
#         88 boolean0_true VALUE X'01' THROUGH X'FF'.
#
#     01 nullCast PIC X.
#         88 nullCast_false VALUE X'00'.
#         88 nullCast_true VALUE X'01' THROUGH X'FF'.
#
# PROCEDURE DIVISION.
# """
#     func="""
# IDENTIFICATION DIVISION.
# PROGRAM-ID. CLASS_c2a773c670339b0d7be-TEST.
#
# ENVIRONMENT DIVISION.
#
# DATA DIVISION.
# LINKAGE SECTION.
#     01 loopIdx PIC S9(9).
#     01 boolean0 PIC X.
#         88 boolean0_false VALUE X'00'.
#         88 boolean0_true VALUE X'01' THROUGH X'FF'.
#
#     01 nullCast PIC X(100).
#
# PROCEDURE DIVISION.
# """
#     p = CobolProcessor()
#     tokens = " ".join(p.tokenize_code(func))
#     inputs = p.extract_arguments(tokens)
    pass