# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( dist ) :
    count = [ 0 ] * ( dist + 1 )
    count [ 0 ] = 1
    count [ 1 ] = 1
    count [ 2 ] = 2
    for i in range ( 3 , dist + 1 ) :
        count [ i ] = ( count [ i - 1 ] + count [ i - 2 ] + count [ i - 3 ] )
    return count [ dist ] ;


#TOFILL

if __name__ == '__main__':
    param = [
    (37,),
    (82,),
    (87,),
    (80,),
    (92,),
    (58,),
    (98,),
    (53,),
    (11,),
    (58,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))