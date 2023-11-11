# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( n ) :
    count = 0 ;
    curr = 19 ;
    while ( True ) :
        sum = 0 ;
        x = curr ;
        while ( x > 0 ) :
            sum = sum + x % 10 ;
            x = int ( x / 10 ) ;
        if ( sum == 10 ) :
            count += 1 ;
        if ( count == n ) :
            return curr ;
        curr += 9 ;
    return - 1 ;


#TOFILL

if __name__ == '__main__':
    param = [
    (93,),
    (10,),
    (55,),
    (94,),
    (2,),
    (5,),
    (37,),
    (4,),
    (11,),
    (46,)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))