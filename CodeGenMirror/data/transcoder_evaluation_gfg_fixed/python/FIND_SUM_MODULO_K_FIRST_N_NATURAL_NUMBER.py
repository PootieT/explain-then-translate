# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( N , K ) :
    ans = 0 ;
    for i in range ( 1 , N + 1 ) :
        ans += ( i % K ) ;
    return ans ;


#TOFILL

if __name__ == '__main__':
    param = [
    (11,5,),
    (36,69,),
    (71,28,),
    (74,1,),
    (66,84,),
    (38,14,),
    (2,11,),
    (73,87,),
    (79,11,),
    (30,55,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))