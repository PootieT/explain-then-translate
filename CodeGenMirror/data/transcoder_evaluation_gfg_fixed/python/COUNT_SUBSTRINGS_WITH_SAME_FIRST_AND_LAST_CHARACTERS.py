# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( s ) :
    result = 0 ;
    n = len ( s ) ;
    for i in range ( n ) :
        for j in range ( i , n ) :
            if ( s [ i ] == s [ j ] ) :
                result = result + 1
    return result


#TOFILL

if __name__ == '__main__':
    param = [
    ('LZIKA',),
    ('0556979952',),
    ('110010',),
    ('kGaYfd',),
    ('413567670657',),
    ('01001',),
    ('EQPuFa',),
    ('48848378',),
    ('110',),
    ('PLehNeP',)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))