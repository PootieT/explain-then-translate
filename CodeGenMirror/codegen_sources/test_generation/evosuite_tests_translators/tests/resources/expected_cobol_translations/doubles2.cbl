IDENTIFICATION DIVISION.
PROGRAM-ID. CLASS_002b132ad75cae1a61c-TEST.

ENVIRONMENT DIVISION.

DATA DIVISION.
WORKING-STORAGE SECTION.
    01 loopIdx PIC S9(9).
    01 test0double0 USAGE COMP-2.
    01 test0argDouble USAGE COMP-2.
    01 test1double0 USAGE COMP-2.
    01 test0argDouble0 USAGE COMP-2.
    01 test3double0 USAGE COMP-2.
    01 test0argDouble1 USAGE COMP-2.




PROCEDURE DIVISION.
Begin.
    PERFORM test0.
    PERFORM test1.
    PERFORM test3.
    STOP RUN.

test0.
    MOVE 32.0 TO test0argDouble.
    CALL "f_filled" USING test0argDouble, test0double0 END-CALL.
    IF NOT FUNCTION ABS(0.0 -  double0) <=  1.0E-4 THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test1.
    MOVE 20873.386 TO test0argDouble0.
    CALL "f_filled" USING test0argDouble0, test1double0 END-CALL.
    IF NOT FUNCTION ABS(-20841.386 -  double0) <=  1.0E-4 THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test3.
    MOVE 0.0 TO test0argDouble1.
    CALL "f_filled" USING test0argDouble1, test3double0 END-CALL.
    IF NOT FUNCTION ABS(32.0 -  double0) <=  1.0E-4 THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.
