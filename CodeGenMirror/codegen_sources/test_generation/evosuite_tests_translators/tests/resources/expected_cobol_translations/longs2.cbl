IDENTIFICATION DIVISION.
PROGRAM-ID. CLASS_0156e303db12f3fac41-TEST.

ENVIRONMENT DIVISION.

DATA DIVISION.
WORKING-STORAGE SECTION.
    01 loopIdx PIC S9(9).
    01 long0 PIC S9(18) COMP.
    01 argLong PIC S9(18) COMP.
    01 argLong0 PIC S9(18) COMP.
    01 argLong1 PIC S9(18) COMP.
    01 argLong2 PIC S9(18) COMP.
    01 argLong3 PIC S9(18) COMP.
    01 argLong4 PIC S9(18) COMP.




PROCEDURE DIVISION.
Begin.
    PERFORM test0.
    PERFORM test1.
    PERFORM test2.
    STOP RUN.

test0.
    MOVE 1 TO argLong.
    MOVE -74133 TO argLong0.
    CALL "f_filled" USING argLong, argLong0, long0 END-CALL.
    IF NOT (148266 EQUALS  long0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test1.
    MOVE 760 TO argLong1.
    MOVE 760 TO argLong2.
    CALL "f_filled" USING argLong1, argLong2, long0 END-CALL.
    IF NOT (-578360 EQUALS  long0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test2.
    MOVE -1 TO argLong3.
    MOVE -1 TO argLong4.
    CALL "f_filled" USING argLong3, argLong4, long0 END-CALL.
    IF NOT (0 EQUALS  long0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.
