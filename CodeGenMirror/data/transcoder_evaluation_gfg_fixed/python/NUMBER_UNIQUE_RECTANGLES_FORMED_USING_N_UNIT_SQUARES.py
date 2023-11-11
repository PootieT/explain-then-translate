# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

def f_gold ( n ) :
    ans = 0
    for length in range ( 1 , int ( math.sqrt ( n ) ) + 1 ) :
        height = length
        while ( height * length <= n ) :
            ans += 1
            height += 1
    return ans


#TOFILL

if __name__ == '__main__':
    param = [
    (34,),
    (49,),
    (41,),
    (17,),
    (67,),
    (38,),
    (59,),
    (64,),
    (61,),
    (58,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))