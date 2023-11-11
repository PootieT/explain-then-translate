IDENTIFICATION DIVISION.
PROGRAM-ID. CLASS_e045b894a398fa5a7c3-TEST.

ENVIRONMENT DIVISION.

DATA DIVISION.
WORKING-STORAGE SECTION.
    01 loopIdx PIC S9(9).
    01 test0long0 PIC S9(18) COMP.
    01 test0argInt PIC S9(9) COMP.
    01 test0argInt0 PIC S9(9) COMP.
    01 test1long0 PIC S9(18) COMP.
    01 test0argInt1 PIC S9(9) COMP.
    01 test0argInt2 PIC S9(9) COMP.
    01 test2long0 PIC S9(18) COMP.
    01 test0argInt3 PIC S9(9) COMP.
    01 test0argInt4 PIC S9(9) COMP.
    01 test2long00 PIC S9(18) COMP.
    01 test0argInt5 PIC S9(9) COMP.
    01 test0argInt6 PIC S9(9) COMP.
    01 test0longArray0_table.
        02 test0longArray0 PIC S9(18) COMP OCCURS 100.
    01 test1longArray0_table.
        02 test1longArray0 PIC S9(18) COMP OCCURS 100.
    01 test2longArray0_table.
        02 test2longArray0 PIC S9(18) COMP OCCURS 100.
    01 test3longArray0_table.
        02 test3longArray0 PIC S9(18) COMP OCCURS 100.




PROCEDURE DIVISION.
Begin.
    PERFORM test0.
    PERFORM test1.
    PERFORM test2.
    PERFORM test3.
    STOP RUN.

test0.
    MOVE 0 TO test0argInt.
    MOVE 0 TO test0argInt0.
    CALL "f_filled" USING test0longArray0_table, test0argInt, test0argInt0, test0long0 END-CALL.
    IF NOT (0 EQUALS  long0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test1.
    MOVE 17676 TO test0argInt1.
    MOVE 0 TO test0argInt2.
    CALL "f_filled" USING test1longArray0_table, test0argInt1, test0argInt2, test1long0 END-CALL.
    IF NOT (0 EQUALS  long0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test2.
    MOVE 60602 TO test2longArray0(1).
    MOVE 0 TO test0argInt3.
    MOVE 3 TO test0argInt4.
    CALL "f_filled" USING test2longArray0_table, test0argInt3, test0argInt4, test2long0 END-CALL.
    IF NOT (60602 EQUALS  long0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test3.
    MOVE -6892 TO test3longArray0(1).
    MOVE 0 TO test0argInt5.
    MOVE 1 TO test0argInt6.
    CALL "f_filled" USING test3longArray0_table, test0argInt5, test0argInt6, test2long00 END-CALL.
    IF NOT (-6892 EQUALS  long0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.
