# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( n ) :
    if ( n == 0 ) :
        return 0
    else :
        return 1 + f_gold ( n & ( n - 1 ) )


#TOFILL

if __name__ == '__main__':
    param = [
    (6,),
    (58,),
    (90,),
    (69,),
    (15,),
    (54,),
    (60,),
    (51,),
    (46,),
    (91,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))