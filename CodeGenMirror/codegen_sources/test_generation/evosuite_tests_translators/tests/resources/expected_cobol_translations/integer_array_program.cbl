IDENTIFICATION DIVISION.
PROGRAM-ID. f_filled.
DATA DIVISION.

WORKING-STORAGE SECTION.
    01 i PIC S9(9) COMP.
    01 temp PIC S9(9) COMP.

LINKAGE SECTION.
    01 a_table.
      02 a PIC S9(9) COMP OCCURS 50.
    01 b_table.
      02 b PIC S9(9) COMP OCCURS 50.
    01 n PIC S9(9) COMP.
    01 k PIC S9(9) COMP.
    01 result PIC X.
        88 result_false VALUE X'00'.
        88 result_true VALUE X'01' THROUGH X'FF'.

PROCEDURE DIVISION USING a_table, b_table, n, k, result.
    *>PERFORM VARYING i FROM 1 BY 1 UNTIL i < n
    *>    IF a(i) + b(i) < k
    *>        SET result_false TO TRUE
    *>        STOP RUN
    *>    END-IF
    *>END-PERFORM
    SET result_true TO TRUE.
    *>SET result_false TO TRUE.
    *>STOP RUN.
