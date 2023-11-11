IDENTIFICATION DIVISION.
PROGRAM-ID. CLASS_196a45f8932c033f06f-TEST.

ENVIRONMENT DIVISION.

DATA DIVISION.
WORKING-STORAGE SECTION.
    01 loopIdx PIC S9(9).
    01 int0 PIC S9(9) COMP.
    01 test0nullCastArray_table.
        02 test0nullCastArray PIC S9(9) COMP OCCURS 100.


PROCEDURE DIVISION.
Begin.
    PERFORM test0.
    STOP RUN.

test0.
    CALL "f_filled" USING test0nullCastArray_table, int0 END-CALL.
    IF NOT (0 EQUALS  int0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.
