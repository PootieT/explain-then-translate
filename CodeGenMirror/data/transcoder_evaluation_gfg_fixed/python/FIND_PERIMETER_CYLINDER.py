# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( diameter , height ) :
    return 2 * ( diameter + height )


#TOFILL

if __name__ == '__main__':
    param = [
    (70,78,),
    (97,46,),
    (49,26,),
    (56,61,),
    (87,79,),
    (64,21,),
    (75,93,),
    (90,16,),
    (55,16,),
    (73,29,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))