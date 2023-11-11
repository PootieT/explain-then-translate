# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( n ) :
    if ( n == 0 or n == 1 ) :
        return 1 ;
    return n * f_gold ( n - 2 ) ;


#TOFILL

if __name__ == '__main__':
    param = [
    (52,),
    (93,),
    (15,),
    (72,),
    (61,),
    (21,),
    (83,),
    (87,),
    (75,),
    (75,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))