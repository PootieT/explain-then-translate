# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( n ) :
    return ( 3 * n * n - n ) // 2


#TOFILL

if __name__ == '__main__':
    param = [
    (96,),
    (93,),
    (15,),
    (8,),
    (21,),
    (14,),
    (11,),
    (79,),
    (24,),
    (94,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))