# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#

import re
from typing import Union, List

from codegen_sources.preprocessing.utils import split_arguments


class EvosuiteTranslator:
    def __init__(self):
        self.primitive_types = {
            "short",
            "int",
            "long",
            "float",
            "double",
            "boolean",
            "char",
        }
        self.java_standard_types = {
            "Double",
            "Float",
            "String",
            "Integer",
            "Boolean",
            "Long",
            "Short",
            "Character",
        }
        self.java_simple_types = self.primitive_types | self.java_standard_types
        self.java_separator_chars = "[^0-9A-Za-z_]"
        self.java_arrays = {f"{t}[]" for t in self.java_simple_types}
        self.java_arrays_regexp = {
            x.replace("[", "\[").replace("]", "\]") for x in self.java_arrays
        }
        self.supported_list_objects = ["ArrayList", "LinkedList", "List"]
        self.java_list_objects = {
            f"{list_object}<{t}>"
            for t in self.java_simple_types
            for list_object in self.supported_list_objects
        }
        self.java_supported_types = (
            self.java_simple_types | self.java_arrays | self.java_list_objects
        )
        self.supported_asserts = [
            "assertTrue",
            "assertFalse",
            "assertEquals",
            "assertArrayEquals",
            "assertNotNull",
            "assertNotSame",
            "assertSame",
            "assertNull",
        ]

        self.assert_argument_extractors = {
            assert_name: re.compile(assert_name + r"\((.+?)\);")
            for assert_name in self.supported_asserts
        }

        # translate_type_castings
        self.type_casting_regexp = {
            t: re.compile(r"\(%s\)[ ]*([^;,\n ?]+)" % t)
            for t in self.java_simple_types
            | self.java_list_objects
            | self.java_arrays_regexp
            | {"Object"}
        }

        # method and class name regexp
        self.method_name_regexp = re.compile(
            "public void test([0-9]+)\(\)  throws Throwable  {"
        )

        # translate_equals
        self.equals_regexp = re.compile(r".equals\(([^;\n]*)\)")

        # translate_value_initializations
        self.double_initialization_regexp = re.compile(
            f"({self.java_separator_chars})([0-9]*\.[0-9E]+)[fFdD]({self.java_separator_chars})"
        )
        self.long_initialization_regexp = re.compile(
            f"({self.java_separator_chars})([0-9]+)[lL]({self.java_separator_chars})"
        )
        self.null_pointers_regexp = re.compile(
            f"({self.java_separator_chars})null({self.java_separator_chars})"
        )

        # translate variable definitions
        self.object_variable_definition = {
            t: re.compile(r"%s ([^=;]+?) = new %s(\([^;]+?\));" % (t, t))
            for t in self.java_standard_types
        }
        self.primitive_variable_definition = {
            t: re.compile(r"%s ([^=;]+?) = ([^;]+?);" % t)
            for t in self.primitive_types | self.java_standard_types
        }

        # array translation
        self.regexp_match_array_content_definition = {
            t: re.compile(r"new %s\[\] \{(.+)\}" % t) for t in self.java_simple_types
        }  # \1 matches the content in the array definition
        self.regexp_match_array_content_definition_line = {
            t: re.compile(r"[\w\[\]<>\s]* ([\w\d]*) = new %s\[\] \{(.+)\};" % t)
            for t in self.java_simple_types
        }  # \1 matches the content in the array definition
        self.regexp_match_array_definition_with_length = {
            t: re.compile(r"%s\[\] (.+?) = new %s\[([0-9]+)\];" % (t, t))
            for t in self.java_simple_types
        }  # \1 is the token identifier and \2 the length of the array
        self.regexp_match_array_length_getter = {
            t: re.compile(r" (([test]*\d+)?%sArray[0-9]+)\.length" % t.lower())
            for t in self.java_simple_types
        }  # \1 is the array identifier name
        ## list translations
        self.list_objects_definitions = {
            t: re.compile(r"%s ([^=;]+?) = new %s\(\);" % (t, t))
            for t in self.java_list_objects
        }
        self.regexp_match_list_definition = {
            simple_type: {
                list_type: re.compile(f"new {list_type}<{simple_type}>\(\)")
                for list_type in self.supported_list_objects
            }
            for simple_type in self.java_simple_types
        }  # \1 matches the content in the array definition
        self.regexp_match_add_to_list = {
            list_type: re.compile(f"({self.type_to_varname(list_type)}[0-9]+)\.add\(")
            for list_type in self.supported_list_objects
        }
        self.regexp_match_list_contains = {
            list_type: re.compile(
                f"({self.type_to_varname(list_type)}[0-9]+)\.contains\(([^\n]*?)\)"
            )
            for list_type in self.supported_list_objects
        }

    def type_to_varname(self, t):
        return t[0].lower() + t[1:]

    def get_asserts_arguments(self, code):
        return {
            assert_name: self.specifics_assert_args(code, assert_name)
            for assert_name in self.supported_asserts
        }

    def specifics_assert_args(self, code, assert_name):
        arguments = set(self.assert_argument_extractors[assert_name].findall(code))
        return [split_arguments(a) for a in set(arguments)]

    @staticmethod
    def replace_func_calls(classname, code):
        return re.sub(
            f"([{classname[0].lower()}, {classname[0]}]{classname[1:]}(0|1|_0)?)"
            + r"\.(.+?)\(",
            r"f_filled(",
            code,
        )

    @staticmethod
    def args_to_string(args_list):
        return ",".join(args_list)

    def get_default_value(self, t):
        if t not in self.primitive_types:
            return "None"
        elif t == "char":
            return "'\u0000'"
        elif t == "boolean":
            return "False"
        else:
            return "0"

    def get_line_containing(self, code, idx: int, delim: str = ";\n") -> str:
        start = self.get_line_start(code, idx, delim)
        end = self.get_line_end(code, idx, delim)
        return code[start:end]

    @staticmethod
    def get_line_start(code, idx: int, delim: str = ";\n") -> int:
        start = idx - 1
        if isinstance(delim, str):
            while code[start : start + len(delim)] != delim and start >= 0:
                start -= 1
            return start
        else:
            delims = delim
            cond = True
            while cond and start >= 0:
                start -= 1
                for delim in delims:
                    cond = cond and code[start : start + len(delim)] != delim
            return start

    @staticmethod
    def get_line_end(code, idx: int, delim: Union[str, List[str]] = ";\n") -> int:
        end = idx
        if isinstance(delim, str):
            while code[end : end + len(delim)] != delim and end < len(code):
                end += 1
            return end
        else:
            delims = delim
            cond = True
            while cond and end < len(code):
                end += 1
                for delim in delims:
                    cond = cond and code[end : end + len(delim)] != delim
            return end
