# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( n ) :
    sum = 0
    for i in range ( n ) :
        sum += i * ( n - i )
    return 2 * sum


#TOFILL

if __name__ == '__main__':
    param = [
    (63,),
    (72,),
    (28,),
    (35,),
    (6,),
    (70,),
    (20,),
    (8,),
    (8,),
    (35,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))