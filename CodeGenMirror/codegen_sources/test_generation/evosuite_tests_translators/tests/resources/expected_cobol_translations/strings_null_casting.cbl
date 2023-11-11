IDENTIFICATION DIVISION.
PROGRAM-ID. CLASS_c2a773c670339b0d7be-TEST.

ENVIRONMENT DIVISION.

DATA DIVISION.
WORKING-STORAGE SECTION.
    01 loopIdx PIC S9(9).
    01 boolean0 PIC X.
        88 boolean0_false VALUE X'00'.
        88 boolean0_true VALUE X'01' THROUGH X'FF'.

    01 nullCast PIC X(100).


PROCEDURE DIVISION.
Begin.
    PERFORM test0.
    STOP RUN.

test0.
    CALL "f_filled" USING "", nullCast, boolean0 END-CALL.
    IF boolean0_true THEN
        DISPLAY 'failure'
        EXIT PARAGRAPH
    END-IF.
    DISPLAY 'success'.
