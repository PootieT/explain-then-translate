# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( n , m ) :
    return n if ( n == m ) else 1


#TOFILL

if __name__ == '__main__':
    param = [
    (57,57,),
    (22,22,),
    (17,17,),
    (74,74,),
    (93,22,),
    (56,54,),
    (5,33,),
    (5,68,),
    (9,75,),
    (98,21,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))