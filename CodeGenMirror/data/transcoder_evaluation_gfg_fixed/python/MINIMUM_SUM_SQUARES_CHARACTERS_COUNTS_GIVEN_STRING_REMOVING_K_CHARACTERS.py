# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( str , k ) :
    l = len ( str )
    if ( k >= l ) :
        return 0
    frequency = [ 0 ] * MAX_CHAR
    for i in range ( 0 , l ) :
        frequency [ ord ( str [ i ] ) - 97 ] += 1
    q = PriorityQueue ( )
    for i in range ( 0 , MAX_CHAR ) :
        q.put ( - frequency [ i ] )
    while ( k > 0 ) :
        temp = q.get ( )
        temp = temp + 1
        q.put ( temp , temp )
        k = k - 1
    result = 0 ;
    while not q.empty ( ) :
        temp = q.get ( )
        temp = temp * ( - 1 )
        result += temp * temp
    return result


#TOFILL

if __name__ == '__main__':
    param = [
    ('JlIgDXBFbCeFRB',88,),
    ('41122661',91,),
    ('011',31,),
    ('hOCcIOAJztdT',83,),
    ('155799263',74,),
    ('1111',35,),
    ('Egy',30,),
    ('6900599415',56,),
    ('101010011111',69,),
    ('IbmRqJcU',21,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))