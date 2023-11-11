IDENTIFICATION DIVISION.
PROGRAM-ID. PERMUTE_TWO_ARRAYS_SUM_EV-TEST.

ENVIRONMENT DIVISION.

DATA DIVISION.
WORKING-STORAGE SECTION.
    01 loopIdx PIC S9(9).
    01 boolean0 PIC X.
        88 boolean0_false VALUE X'00'.
        88 boolean0_true VALUE X'01' THROUGH X'FF'.

    01 test0integerArray0_table.
        02 test0integerArray0 PIC S9(9) COMP OCCURS 2.
    01 test1integerArray0_table.
        02 test1integerArray0 PIC S9(9) COMP OCCURS 2.
    01 test2integerArray0_table.
        02 test2integerArray0 PIC S9(9) COMP OCCURS 2.
    01 test3integerArray0_table.
        02 test3integerArray0 PIC S9(9) COMP OCCURS 2.
    01 test0inlineArray_table.
        02 test0inlineArray PIC S9(9) COMP OCCURS 3.
    01 test1inlineArray_table.
        02 test1inlineArray PIC S9(9) COMP OCCURS 3.
    01 test2inlineArray_table.
        02 test2inlineArray PIC S9(9) COMP OCCURS 3.
    01 test3inlineArray_table.
        02 test3inlineArray PIC S9(9) COMP OCCURS 3.
    01 test0intArray0_table.
        02 test0intArray0 PIC S9(9) COMP OCCURS 3.
    01 test1intArray0_table.
        02 test1intArray0 PIC S9(9) COMP OCCURS 3.
    01 test2intArray0_table.
        02 test2intArray0 PIC S9(9) COMP OCCURS 3.
    01 test3intArray0_table.
        02 test3intArray0 PIC S9(9) COMP OCCURS 3.
    01 integer0 PIC S9(9) COMP.
    01 integer1 PIC S9(9) COMP.
    01 int0 PIC S9(9) COMP.
    01 int1 PIC S9(9) COMP.




PROCEDURE DIVISION.
Begin.
    PERFORM test0.
    PERFORM test1.
    PERFORM test2.
    PERFORM test3.
    STOP RUN.

test0.
    MOVE -1 TO int0.
    MOVE -1 TO integer0.
    IF NOT (-1 EQUALS  integer0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    IF integer0 = (int0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    IF NOT (integer0 NOT = SPACE AND LOW-VALUE) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.

    MOVE integer0 TO test0integerArray0(1).
    MOVE 1 TO integer1.
    IF NOT (1 EQUALS  integer1) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    IF integer1 = (int0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    IF integer1 = (integer0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    IF NOT (integer1 NOT = SPACE AND LOW-VALUE) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.

    MOVE integer1 TO test0integerArray0(2).
    MOVE int0 TO test0intArray0(3).
    CALL "f_filled" USING test0integerArray0_table, test0intArray0_table, 1, 0, boolean0 END-CALL.
    IF boolean0_false THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    MOVE -1 TO test0inlineArray(1).
    MOVE 0 TO test0inlineArray(2).
    MOVE 0 TO test0inlineArray(3).
    PERFORM VARYING loopIdx FROM 1 BY 1 UNTIL loopIdx > 100
        IF NOT test0inlineArray(loopIdx) =  test0intArray0(loopIdx) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF
    END-PERFORM.
    DISPLAY 'success'.

test1.
    MOVE -1 TO int0.
    MOVE -1 TO integer0.
    IF NOT (-1 EQUALS  integer0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    IF integer0 = (int0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    IF NOT (integer0 NOT = SPACE AND LOW-VALUE) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.

    MOVE integer0 TO test1integerArray0(1).
    MOVE 1 TO int1.
    MOVE integer0 TO test1integerArray0(2).
    MOVE int0 TO test1intArray0(3).
    CALL "f_filled" USING test1integerArray0_table, test1intArray0_table, int1, -50146, boolean0 END-CALL.
    IF boolean0_false THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    MOVE -1 TO test1inlineArray(1).
    MOVE 0 TO test1inlineArray(2).
    MOVE 0 TO test1inlineArray(3).
    PERFORM VARYING loopIdx FROM 1 BY 1 UNTIL loopIdx > 100
        IF NOT test1inlineArray(loopIdx) =  test1intArray0(loopIdx) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF
    END-PERFORM.
    IF int1 = int0 THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test2.
    MOVE -1 TO integer0.
    IF NOT (-1 EQUALS  integer0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    IF NOT (integer0 NOT = SPACE AND LOW-VALUE) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.

    MOVE integer0 TO test2integerArray0(1).
    MOVE integer0 TO test2integerArray0(2).
    CALL "f_filled" USING test2integerArray0_table, test2intArray0_table, -54229, 1, boolean0 END-CALL.
    IF boolean0_false THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    MOVE 0 TO test2inlineArray(1).
    MOVE 0 TO test2inlineArray(2).
    MOVE 0 TO test2inlineArray(3).
    PERFORM VARYING loopIdx FROM 1 BY 1 UNTIL loopIdx > 100
        IF NOT test2inlineArray(loopIdx) =  test2intArray0(loopIdx) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF
    END-PERFORM.
    DISPLAY 'success'.

test3.
    MOVE -1 TO integer0.
    IF NOT (-1 EQUALS  integer0) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    IF NOT (integer0 NOT = SPACE AND LOW-VALUE) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.

    MOVE integer0 TO test3integerArray0(1).
    MOVE 1 TO int0.
    MOVE test3integerArray0(1) TO test3integerArray0(2).
    CALL "f_filled" USING test3integerArray0_table, test3intArray0_table, 1, int0, boolean0 END-CALL.
    IF boolean0_true THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    MOVE 0 TO test3inlineArray(1).
    MOVE 0 TO test3inlineArray(2).
    MOVE 0 TO test3inlineArray(3).
    PERFORM VARYING loopIdx FROM 1 BY 1 UNTIL loopIdx > 100
        IF NOT test3inlineArray(loopIdx) =  test3intArray0(loopIdx) THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF
    END-PERFORM.
    DISPLAY 'success'.
