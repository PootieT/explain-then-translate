// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//

import java.util.*;
import java.util.stream.*;
import java.lang.*;
// import javafx.util.Pair;

public class SMALLEST_SUM_CONTIGUOUS_SUBARRAY {
static int f_gold(int arr[], int n) {
  int min_ending_here = 2147483647;
  int min_so_far = 2147483647;
  for (int i = 0; i < n; i++) {
    if (min_ending_here > 0) min_ending_here = arr[i];
    else min_ending_here += arr[i];
    min_so_far = Math.min(min_so_far, min_ending_here);
  }
  return min_so_far;
}

//TOFILL

public static void main(String args[]) {
  int n_success = 0;
  List<int[]> param0 = new ArrayList<>();
  param0.add(
      new int[] {
        2, 9, 13, 14, 15, 18, 19, 19, 25, 26, 29, 29, 29, 30, 31, 36, 37, 37, 38, 39, 39, 40, 40,
        42, 42, 46, 50, 53, 58, 60, 62, 64, 65, 67, 68, 69, 72, 77, 78, 83, 85, 89, 90, 93, 95,
        95, 97
      });
  param0.add(new int[] {14, -58, 8, 78, -26, -20, -60, 42, -64, -12});
  param0.add(new int[] {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  param0.add(
      new int[] {
        44, 88, 20, 47, 69, 42, 26, 49, 71, 91, 18, 95, 9, 66, 60, 37, 47, 29, 98, 63, 15, 9, 80,
        66, 1, 9, 57, 56, 20, 2, 1
      });
  param0.add(new int[] {-78, -64, -62, -60, -52, 4, 8, 46, 72, 74});
  param0.add(new int[] {0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1});
  param0.add(
      new int[] {
        3, 7, 16, 17, 23, 23, 23, 28, 29, 30, 34, 38, 40, 41, 43, 43, 44, 46, 51, 51, 51, 55, 57,
        58, 61, 62, 66, 66, 67, 69, 70, 73, 75, 77, 79, 80, 85, 85, 87, 87, 93, 96
      });
  param0.add(
      new int[] {
        80, 22, 38, 26, 62, -48, -48, 46, -54, 4, 76, 46, 48, 40, -92, -98, -88, 12, -42, -94, 76,
        -16, -82, 62, 18, -24
      });
  param0.add(new int[] {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1});
  param0.add(
      new int[] {
        85, 44, 1, 97, 50, 74, 62, 90, 3, 78, 8, 48, 96, 41, 36, 91, 57, 97, 85, 90, 78, 43, 28,
        92, 85, 59, 29, 38, 34, 65, 20, 26, 27, 23, 71, 22, 47, 99, 68, 93, 67, 66, 69, 82, 98,
        15, 44, 51, 65
      });
  List<Integer> param1 = new ArrayList<>();
  param1.add(24);
  param1.add(6);
  param1.add(8);
  param1.add(26);
  param1.add(8);
  param1.add(11);
  param1.add(38);
  param1.add(22);
  param1.add(13);
  param1.add(45);
  for (int i = 0; i < param0.size(); ++i) {
    if (f_filled(param0.get(i), param1.get(i)) == f_gold(param0.get(i), param1.get(i))) {
      n_success += 1;
    }
  }
  System.out.println("#Results:" + n_success + ", " + param0.size());
}}