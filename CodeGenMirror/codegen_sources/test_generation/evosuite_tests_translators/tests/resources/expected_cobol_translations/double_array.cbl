IDENTIFICATION DIVISION.
PROGRAM-ID. CLASS_005ae0a2dee4fd5b484-TEST.

ENVIRONMENT DIVISION.

DATA DIVISION.
WORKING-STORAGE SECTION.
    01 loopIdx PIC S9(9).
    01 test0double0 USAGE COMP-2.
    01 test1double0 USAGE COMP-2.
    01 test2double0 USAGE COMP-2.
    01 test3double0 USAGE COMP-2.
    01 test4double0 USAGE COMP-2.
    01 test6double0 USAGE COMP-2.
    01 test7double2 USAGE COMP-2.
    01 test8double1 USAGE COMP-2.
    01 test9double1 USAGE COMP-2.
    01 test6doubleArray0_table.
        02 test6doubleArray0 USAGE COMP-2 OCCURS 100.
    01 test7doubleArray0_table.
        02 test7doubleArray0 USAGE COMP-2 OCCURS 100.
    01 test8doubleArray0_table.
        02 test8doubleArray0 USAGE COMP-2 OCCURS 100.
    01 test8doubleArray1_table.
        02 test8doubleArray1 USAGE COMP-2 OCCURS 100.
    01 test9doubleArray0_table.
        02 test9doubleArray0 USAGE COMP-2 OCCURS 100.
    01 test0test6doubleArray0_table.
        02 test0test6doubleArray0 USAGE COMP-2 OCCURS 100.
    01 test1test6doubleArray0_table.
        02 test1test6doubleArray0 USAGE COMP-2 OCCURS 100.
    01 test2test6doubleArray0_table.
        02 test2test6doubleArray0 USAGE COMP-2 OCCURS 100.
    01 test3test6doubleArray0_table.
        02 test3test6doubleArray0 USAGE COMP-2 OCCURS 100.
    01 test4test6doubleArray0_table.
        02 test4test6doubleArray0 USAGE COMP-2 OCCURS 100.
    01 double0 USAGE COMP-2.




PROCEDURE DIVISION.
Begin.
    PERFORM test0.
    PERFORM test1.
    PERFORM test2.
    PERFORM test3.
    PERFORM test4.
    PERFORM test6.
    PERFORM test7.
    PERFORM test8.
    PERFORM test9.
    STOP RUN.

test0.
    MOVE -1.0 TO test0test6doubleArray0(1).
    CALL "f_filled" USING test0test6doubleArray0_table, test0double0 END-CALL.
    CANCEL "f_filled".
    IF NOT FUNCTION ABS(-0.14285714285714285 -  test0double0) <=  1.0E-4 THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test1.
    CALL "f_filled" USING test1test6doubleArray0_table, test1double0 END-CALL.
    CANCEL "f_filled".
    IF NOT FUNCTION ABS(0.0 -  test1double0) <=  1.0E-4 THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test2.
    CALL "f_filled" USING test2test6doubleArray0_table, test2double0 END-CALL.
    CANCEL "f_filled".
    IF NOT FUNCTION ABS(0.0 -  test2double0) <=  1.0E-4 THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test3.
    MOVE -56615.2289 TO test3test6doubleArray0(1).
    CALL "f_filled" USING test3test6doubleArray0_table, test3double0 END-CALL.
    CANCEL "f_filled".
    IF NOT FUNCTION ABS(-56615.2289 -  test3double0) <=  1.0E-4 THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test4.
    MOVE 19833.7892 TO test4test6doubleArray0(1).
    CALL "f_filled" USING test4test6doubleArray0_table, test4double0 END-CALL.
    CANCEL "f_filled".
    IF NOT FUNCTION ABS(19833.7892 -  test4double0) <=  1.0E-4 THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test6.
    CALL "f_filled" USING test7doubleArray0_table, test6double0 END-CALL.
    CANCEL "f_filled".
    IF NOT FUNCTION ABS(0.0 -  test6double0) <=  1.0E-4 THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test7.
    MOVE -1.0 TO double0.
    MOVE double0 TO test7doubleArray0(1).
    MOVE 91376.784102651 TO test8double1.
    MOVE test8double1 TO test7doubleArray0(2).
    MOVE double0 TO test7doubleArray0(3).
    MOVE double0 TO test7doubleArray0(4).
    CALL "f_filled" USING test8doubleArray0_table, test7double2 END-CALL.
    CANCEL "f_filled".
    IF NOT FUNCTION ABS(91373.784102651 -  test7double2) <=  1.0E-4 THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test8.
    MOVE -1.0 TO double0.
    MOVE double0 TO test8doubleArray0(1).
    MOVE double0 TO test8doubleArray1(1).
    MOVE test9doubleArray0(1) TO test8doubleArray1(2).
    CALL "f_filled" USING test8doubleArray1_table, test8double1 END-CALL.
    CANCEL "f_filled".
    IF NOT FUNCTION ABS(-2.0 -  test9double1) <=  1.0E-4 THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test9.
    MOVE 0.0 TO double0.
    MOVE double0 TO test9doubleArray0(1).
    CALL "f_filled" USING doubleArray0_table, test9double1 END-CALL.
    CANCEL "f_filled".
    IF NOT FUNCTION ABS(0.0 -  double1) <=  1.0E-4 THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.

    DISPLAY 'success'.
