from codegen_sources.preprocessing.tests.obfuscation.utils import diff_tester
from codegen_sources.preprocessing.lang_processors.cobol_processor import (
    CobolProcessor,
)

processor = CobolProcessor()


def test_obfuscation_var_definition():
#     input_program = """import os
# class Factorial:
#     def factorial(self, n, path):
#         res, res2, res3 = 1, 1, 1
#         for i in range(n):
#             res *= (i + 1)
#         with open(os.path.join(path, 'res'), 'w') as f:
#             f.write(str(res))
#         return res
#         """
    input_program = """IDENTIFICATION DIVISION.
PROGRAM-ID. clampLong.
DATA DIVISION.
WORKING-STORAGE SECTION.
    

LINKAGE SECTION.
    01 valueLong PIC S9(18) COMP.
    01 min PIC S9(18) COMP.
    01 max PIC S9(18) COMP.
    01 result PIC S9(18) COMP.

PROCEDURE DIVISION USING valueLong, min, max, result.
    
    IF valueLong <= min
        MOVE min TO result
    ELSE IF valueLong >= max
        MOVE max TO result
    ELSE
        MOVE valueLong TO result
    END-IF
    
    GOBACK. 

END PROGRAM clampLong.
    """
    res, dico = processor.obfuscate_code(input_program)
    expected = """IDENTIFICATION DIVISION . PROGRAM-ID . FUNC_0 . DATA DIVISION . WORKING-STORAGE SECTION . LINKAGE SECTION . 01  VAR_0 PIC S9(18) COMP . 01  min PIC S9(18) COMP . 01  max PIC S9(18) COMP . 01  VAR_1 PIC S9(18) COMP . PROCEDURE DIVISION USING VAR_0 , min , max , VAR_1 . IF VAR_0 <= min
        MOVE min TO VAR_1 ELSE IF VAR_0 >= max
        MOVE max TO VAR_1 ELSE
        MOVE VAR_0 TO VAR_1 END-IF
    
    GOBACK . END PROGRAM FUNC_0 ."""
    diff_tester(expected.strip(), res.strip())
    # diff_tester(
    #     "FUNC_0 clampLong | VAR_0 valueLong | VAR_1 result",
    #     dico,
    #     split=" | ",
    # )


def test_obfuscation_recursive_method():
#     input_program = """class Factorial:
#     def factorial(self, n):
#         if n == 1:
#             return 1
#         return n * self.factorial(n-1)
# """
    input_program = """IDENTIFICATION DIVISION.
PROGRAM-ID. factorial.
DATA DIVISION.
WORKING-STORAGE SECTION.

LINKAGE SECTION.
    01 n PIC S9(18) COMP.
    01 result PIC S9(18) COMP.

PROCEDURE DIVISION USING n, result.
    IF n <= 1
        MOVE 1 TO result
    ELSE
        n = n - 1
        CALL "factorial" USING n, result END-CALL.
        MOVE result * n TO result
    END-IF
    GOBACK. 
END PROGRAM factorial.
    """
    res, dico = processor.obfuscate_code(input_program)
    expected = """IDENTIFICATION DIVISION . PROGRAM-ID . FUNC_0 . DATA DIVISION . WORKING-STORAGE SECTION . LINKAGE SECTION . 01  VAR_0 PIC S9(18) COMP . 01  VAR_1 PIC S9(18) COMP . PROCEDURE DIVISION USING VAR_0 , VAR_1 . IF VAR_0 <= 1
         MOVE 1  TO VAR_1 ELSE
        VAR_0 = VAR_0 - 1
         CALL  "FUNC_0" USING VAR_0 , VAR_1 END-CALL . MOVE VAR_1 * VAR_0 TO VAR_1 END-IF
    GOBACK . END PROGRAM FUNC_0 ."""
    diff_tester(expected.strip(), res.strip())
    # diff_tester(
    #     "CLASS_0 Factorial | FUNC_0 factorial | VAR_0 self | VAR_1 n", dico, split=" | "
    # )


if __name__ == "__main__":
    # test_obfuscation_var_definition()
    test_obfuscation_recursive_method()