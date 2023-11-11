# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( A , B , C , M ) :
    res = pow ( B , C )
    ans = pow ( A , res )
    return ans


#TOFILL

if __name__ == '__main__':
    param = [
    (49,7,63,7,),
    (31,75,82,61,),
    (29,10,82,15,),
    (26,59,12,13,),
    (10,76,14,53,),
    (18,40,71,78,),
    (48,21,41,91,),
    (17,35,78,80,),
    (49,73,69,86,),
    (22,85,6,8,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))