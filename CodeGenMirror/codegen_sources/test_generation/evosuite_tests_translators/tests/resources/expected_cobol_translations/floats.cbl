IDENTIFICATION DIVISION.
PROGRAM-ID. CLASS_9167f62308cfc555ab3-TEST.

ENVIRONMENT DIVISION.

DATA DIVISION.
WORKING-STORAGE SECTION.
    01 loopIdx PIC S9(9).
    01 float0 USAGE COMP-1.
    01 argFloat USAGE COMP-1.




PROCEDURE DIVISION.
Begin.
    PERFORM test0.
    PERFORM test1.
    STOP RUN.

test0.
    MOVE 31337.701 TO argFloat.
    CALL "f_filled" USING argFloat, float0 END-CALL.
    IF NOT FUNCTION ABS(9.8205152E8 -  float0) <=  0.01 THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test1.
    MOVE 0.0 TO argFloat.
    CALL "f_filled" USING argFloat, float0 END-CALL.
    IF NOT FUNCTION ABS(0.0 -  float0) <=  0.01 THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.
