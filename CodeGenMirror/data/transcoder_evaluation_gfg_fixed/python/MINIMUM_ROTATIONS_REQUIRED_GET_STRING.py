# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( str ) :
    tmp = str + str
    n = len ( str )
    for i in range ( 1 , n + 1 ) :
        substring = tmp [ i : i+n ]
        if ( str == substring ) :
            return i
    return n


#TOFILL

if __name__ == '__main__':
    param = [
    ('vdevdNdQSopPtj',),
    ('5',),
    ('100010101011',),
    ('tlDOvJHAyMllu',),
    ('06',),
    ('101',),
    ('DYgtU',),
    ('4',),
    ('00',),
    ('Dt',)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))