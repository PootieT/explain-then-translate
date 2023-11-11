# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( s ) :
    n = len ( s )
    count = 0 ;
    for i in range ( 0 , n , 1 ) :
        if ( s [ i ] == '4' or s [ i ] == '8' or s [ i ] == '0' ) :
            count += 1
    for i in range ( 0 , n - 1 , 1 ) :
        h = ( ord ( s [ i ] ) - ord ( '0' ) ) * 10 + ( ord ( s [ i + 1 ] ) - ord ( '0' ) )
        if ( h % 4 == 0 ) :
            count = count + i + 1
    return count


#TOFILL

if __name__ == '__main__':
    param = [
    # ('Qaq',),
    ('9400761825850',),
    ('0011001111',),
    # ('lasWqrLRq',),
    ('5662',),
    ('110',),
    # (' tOYKf',),
    ('6536991235305',),
    ('11111',),
    # ('uZftT iDHcYiCt',)
    ('4',),
    ('1001',),
    ('4444444444444445',),
    ('2035812304128',),
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))