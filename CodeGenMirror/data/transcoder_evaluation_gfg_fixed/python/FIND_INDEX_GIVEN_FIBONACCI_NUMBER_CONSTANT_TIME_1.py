# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

def f_gold ( n ) :
    fibo = 2.078087 * math.log ( n ) + 1.672276
    return round ( fibo )


#TOFILL

if __name__ == '__main__':
    param = [
    (20,),
    (95,),
    (39,),
    (21,),
    (94,),
    (79,),
    (56,),
    (62,),
    (23,),
    (3,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))