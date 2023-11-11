# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( num ) :
    series = [ 1 , 3 , 2 , - 1 , - 3 , - 2 ] ;
    series_index = 0 ;
    result = 0 ;
    for i in range ( ( len ( num ) - 1 ) , - 1 , - 1 ) :
        digit = ord ( num [ i ] ) - 48 ;
        result += digit * series [ series_index ] ;
        series_index = ( series_index + 1 ) % 6 ;
        result %= 7 ;
    if ( result < 0 ) :
        result = ( result + 7 ) % 7 ;
    return result ;


#TOFILL

if __name__ == '__main__':
    param = [
    # ('K',),
    ('850076',),
    ('00111',),
    # ('X',),
    ('1',),
    ('10010001100',),
    # (' pgPeLz',),
    ('53212456821275',),
    ('101',),
    # ('V',)
    ('0',),
    ('777777777777777',),
    ('777777777777776',),
    ('01249012310238510201',)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))