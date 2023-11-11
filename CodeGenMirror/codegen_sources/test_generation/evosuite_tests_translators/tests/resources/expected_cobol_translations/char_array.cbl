IDENTIFICATION DIVISION.
PROGRAM-ID. CLASS_02354123ff83fb6cc72-TEST.

ENVIRONMENT DIVISION.

DATA DIVISION.
WORKING-STORAGE SECTION.
    01 loopIdx PIC S9(9).
    01 string0 PIC X(100).
    01 test1nullCastArray_table.
        02 test1nullCastArray PIC N USAGE NATIONAL OCCURS 100.




PROCEDURE DIVISION.
Begin.
    PERFORM test1.
    STOP RUN.

test1.
    CALL "f_filled" USING test1nullCastArray_table, string0 END-CALL.
    IF (string0 NOT = SPACE AND LOW-VALUE) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.
