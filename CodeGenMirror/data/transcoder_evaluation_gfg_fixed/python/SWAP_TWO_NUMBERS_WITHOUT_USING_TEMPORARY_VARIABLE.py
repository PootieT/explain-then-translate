# Copyright _c_ 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def f_gold ( xp , yp ) :
    xp [ 0 ] = xp [ 0 ] ^ yp [ 0 ]
    yp [ 0 ] = xp [ 0 ] ^ yp [ 0 ]
    xp [ 0 ] = xp [ 0 ] ^ yp [ 0 ]


#TOFILL

if __name__ == '__main__':
    param = [
    ([2, 7, 12, 13, 15, 17, 24, 27, 28, 31, 36, 44, 55, 55, 56, 58, 60, 62, 64, 73, 75, 77, 89, 90, 93, 93, 95, 97, 98],[5, 8, 12, 13, 14, 20, 23, 25, 27, 28, 31, 33, 33, 37, 38, 39, 42, 42, 43, 47, 52, 54, 62, 67, 71, 72, 73, 76, 77, 79, 81, 81, 85, 86, 89, 91, 91, 96, 96, 99],),
    ([36, 51, 6, 25, 13, -36, 23, 14, -80, -84, 45, -81, 20, 36, 66, -62, 81, -7, 5, 0, -10, 59, -56, 88, -24, 45, -21, -27, 67, 70, 76, 40, -4, -11, 3, 46, -94, 22, 71, 16, 63, 82, -38, 97, 44, 36, -11, -4],[-14, 59, 38, 84, 21, -24, -8, -30, 48, 33, -77, 37, -42, 87, 82, -54, -78, 92, -92, 23, -80, 79, -22, -61, -63, 78],),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],),
    ([63, 87, 32, 10, 11, 70, 62, 47, 34, 17, 54, 76, 85, 40, 19, 67, 82, 53, 64, 83, 47, 6, 16, 72, 18, 52, 48, 43, 39, 49, 12, 10, 71, 55, 1, 49, 36, 21, 37, 24, 55, 25, 7, 81, 93, 94, 71, 97, 71],[11, 73, 12, 35, 63, 68, 92, 57, 13, 23, 83, 39, 88, 80, 97, 12, 34, 71, 75, 31, 93, 62, 13, 31],),
    ([-85, 3],[-84, -71, -66, -63, -25, -23, -19, 3, 67, 94, 96],),
    ([1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1],[0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],),
    ([2, 3, 4, 5, 6, 8, 14, 16, 16, 17, 19, 21, 21, 25, 26, 26, 29, 33, 36, 39, 46, 53, 55, 56, 66, 77, 81, 84, 84, 86, 86, 88, 89, 89, 92, 97],[5, 13, 22, 24, 27, 33, 46, 67, 74, 77, 78, 82, 86, 90, 91],),
    ([15, -37, 13, -6, -52, 91, -88, 56, 72, 71, 45, -60, 77, -15, -43, -13, -16, 92, 85, 21, -26, 7, 24, 98, 0, -57, 19, -77, 55, 86, -63, 34, -35, 34, 78, -78, 5, -12, 53, 25, -21, -10, -49, 76, 2, 98, 6],[-40],),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],),
    ([75, 24, 60, 95, 55],[80, 24, 77, 24, 54, 26, 66, 80, 38, 65, 75, 51, 35],)
        ]
    filled_function_param = [
    ([2, 7, 12, 13, 15, 17, 24, 27, 28, 31, 36, 44, 55, 55, 56, 58, 60, 62, 64, 73, 75, 77, 89, 90, 93, 93, 95, 97, 98],[5, 8, 12, 13, 14, 20, 23, 25, 27, 28, 31, 33, 33, 37, 38, 39, 42, 42, 43, 47, 52, 54, 62, 67, 71, 72, 73, 76, 77, 79, 81, 81, 85, 86, 89, 91, 91, 96, 96, 99],),
    ([36, 51, 6, 25, 13, -36, 23, 14, -80, -84, 45, -81, 20, 36, 66, -62, 81, -7, 5, 0, -10, 59, -56, 88, -24, 45, -21, -27, 67, 70, 76, 40, -4, -11, 3, 46, -94, 22, 71, 16, 63, 82, -38, 97, 44, 36, -11, -4],[-14, 59, 38, 84, 21, -24, -8, -30, 48, 33, -77, 37, -42, 87, 82, -54, -78, 92, -92, 23, -80, 79, -22, -61, -63, 78],),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],),
    ([63, 87, 32, 10, 11, 70, 62, 47, 34, 17, 54, 76, 85, 40, 19, 67, 82, 53, 64, 83, 47, 6, 16, 72, 18, 52, 48, 43, 39, 49, 12, 10, 71, 55, 1, 49, 36, 21, 37, 24, 55, 25, 7, 81, 93, 94, 71, 97, 71],[11, 73, 12, 35, 63, 68, 92, 57, 13, 23, 83, 39, 88, 80, 97, 12, 34, 71, 75, 31, 93, 62, 13, 31],),
    ([-85, 3],[-84, -71, -66, -63, -25, -23, -19, 3, 67, 94, 96],),
    ([1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1],[0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],),
    ([2, 3, 4, 5, 6, 8, 14, 16, 16, 17, 19, 21, 21, 25, 26, 26, 29, 33, 36, 39, 46, 53, 55, 56, 66, 77, 81, 84, 84, 86, 86, 88, 89, 89, 92, 97],[5, 13, 22, 24, 27, 33, 46, 67, 74, 77, 78, 82, 86, 90, 91],),
    ([15, -37, 13, -6, -52, 91, -88, 56, 72, 71, 45, -60, 77, -15, -43, -13, -16, 92, 85, 21, -26, 7, 24, 98, 0, -57, 19, -77, 55, 86, -63, 34, -35, 34, 78, -78, 5, -12, 53, 25, -21, -10, -49, 76, 2, 98, 6],[-40],),
    ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],),
    ([75, 24, 60, 95, 55],[80, 24, 77, 24, 54, 26, 66, 80, 38, 65, 75, 51, 35],)
        ]
    n_success = 0
    for i, parameters_set in enumerate(param):
        f_filled(*(filled_function_param[i]))
        f_gold(*parameters_set)
        if parameters_set == filled_function_param[i]:
            n_success+=1
    print("#Results: %i, %i" % (n_success, len(param)))