# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( str ) :
    n = len ( str ) ;
    return int ( n * ( n + 1 ) / 2 ) ;


#TOFILL

if __name__ == '__main__':
    param = [
    ('gZFGZsHCimLf',),
    ('505357',),
    ('011011101',),
    ('ovfwP Osauz',),
    ('92132238746026',),
    ('01100',),
    ('RaOWYQRfiWKSyC',),
    ('861330202',),
    ('001100010',),
    ('uvpKlGUBLOMba',)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        if abs(1 - (0.0000001 + abs(f_gold(*parameters_set))) / (abs(f_filled(*parameters_set)) + 0.0000001)) < 0.001:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))