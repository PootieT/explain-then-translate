# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( a , b ) :
    if a == 0 :
        return b
    return f_gold ( b % a , a )


#TOFILL

if __name__ == '__main__':
    param = [
    (46,89,),
    (26,82,),
    (40,12,),
    (58,4,),
    (25,44,),
    (2,87,),
    (8,65,),
    (21,87,),
    (82,10,),
    (17,61,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))