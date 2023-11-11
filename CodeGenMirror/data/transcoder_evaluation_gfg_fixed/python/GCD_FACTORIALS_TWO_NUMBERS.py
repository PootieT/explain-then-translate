# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

def f_gold ( m , n ) :
    return math.factorial ( min ( m , n ) )


#TOFILL

if __name__ == '__main__':
    param = [
    (81,39,),
    (19,94,),
    (49,37,),
    (40,88,),
    (36,28,),
    (11,46,),
    (82,94,),
    (80,37,),
    (82,75,),
    (56,9,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))