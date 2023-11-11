IDENTIFICATION DIVISION.
PROGRAM-ID. CLASS_023fa3df801cfbc2fb6-TEST.

ENVIRONMENT DIVISION.

DATA DIVISION.
WORKING-STORAGE SECTION.
    01 loopIdx PIC S9(9).
    01 boolean1 PIC X.
        88 boolean1_false VALUE X'00'.
        88 boolean1_true VALUE X'01' THROUGH X'FF'.

    01 boolean0 PIC X.
        88 boolean0_false VALUE X'00'.
        88 boolean0_true VALUE X'01' THROUGH X'FF'.

    01 nullCast PIC X.
        88 nullCast_false VALUE X'00'.
        88 nullCast_true VALUE X'01' THROUGH X'FF'.





PROCEDURE DIVISION.
Begin.
    PERFORM test0.
    PERFORM test1.
    STOP RUN.

test0.
    SET boolean0_true TO true.
    CALL "f_filled" USING boolean0, boolean1 END-CALL.
    IF boolean1_false THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.

test1.
    SET nullCast_false TO true.
    CALL "f_filled" USING nullCast, boolean0 END-CALL.
    IF boolean0_true THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.
