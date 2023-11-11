# Copyright (c) 2022-present, IBM, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# from .test_utils import read_inputs, translation_testing
# from ..evosuite_to_cobol import EvosuiteToCobol

# local testing only
from test_utils import read_inputs, translation_testing
from evosuite_tests_translators.evosuite_to_cobol import EvosuiteToCobol


ARRAYS = ["integer_array_test", "integer_array_casting"]

JAVA_ARRAYS = ["java_list"]

TEST_STRINGS = ["strings", "strings_null_casting"]

TEST_FLOATS = [
    "floats",
    "doubles",
]

TEST_EXTRAS = [
    # "longs",
    # "doubles2",
    "double_array",
    # "longs2",
    # "char_array",
    # "booleans",
]

translator = EvosuiteToCobol()


def test_array_translation():
    translations_list = [read_inputs(filename, "cobol") for filename in ARRAYS]
    translation_testing(translations_list, translator, True)


def test_lists_translation():
    translations_list = [read_inputs(filename, "cobol") for filename in JAVA_ARRAYS]
    translation_testing(translations_list, translator, True)


def test_floats():
    translations_list = [read_inputs(filename, "cobol") for filename in TEST_FLOATS]
    translation_testing(translations_list, translator, True)


def test_string_translation():
    translations_list = [read_inputs(filename, "cobol") for filename in TEST_STRINGS]
    translation_testing(translations_list, translator, True)


def test_extras():
    translations_list = [read_inputs(filename, "cobol") for filename in TEST_EXTRAS]
    translation_testing(translations_list, translator, True)


if __name__ == "__main__":
    # test_floats()  # passes
    # test_array_translation()  # passes
    # test_lists_translation()  # example of untranslatable test, basically empty output
    # test_string_translation()
    test_extras()
