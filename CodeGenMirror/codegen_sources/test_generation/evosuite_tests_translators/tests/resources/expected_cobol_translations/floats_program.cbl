IDENTIFICATION DIVISION.
PROGRAM-ID. f_filled.
DATA DIVISION.
WORKING-STORAGE SECTION.
LINKAGE SECTION.
01 A USAGE COMP-1.
01 B USAGE COMP-1.
PROCEDURE DIVISION USING A, B.
begin.
    MOVE A TO B.
end program f_filled.
