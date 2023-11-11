// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//

import java.util.*;
import java.util.stream.*;
import java.lang.*;
// import javafx.util.Pair;

public class FIND_NUMBER_ENDLESS_POINTS {
static int f_gold(boolean input[][], int n) {
  boolean row[][] = new boolean[n][n];
  boolean col[][] = new boolean[n][n];
  for (int j = 0; j < n; j++) {
    boolean isEndless = true;
    for (int i = n - 1; i >= 0; i--) {
      if (input[i][j] == false) isEndless = false;
      col[i][j] = isEndless;
    }
  }
  for (int i = 0; i < n; i++) {
    boolean isEndless = true;
    for (int j = n - 1; j >= 0; j--) {
      if (input[i][j] == false) isEndless = false;
      row[i][j] = isEndless;
    }
  }
  int ans = 0;
  for (int i = 0; i < n; i++) for (int j = 1; j < n; j++) if (row[i][j] && col[i][j]) ans++;
  return ans;
}

//TOFILL

public static void main(String args[]) {
  int n_success = 0;
  List<boolean[][]> param0 = new ArrayList<>();
  param0.add(
      new boolean[][] {
        new boolean[] {false, false, false, true},
        new boolean[] {false, true, true, true},
        new boolean[] {false, false, true, true},
        new boolean[] {true, true, true, true}
      });
  param0.add(
      new boolean[][] {
        new boolean[] {true, false, true, true, true, true, false, false, false},
        new boolean[] {false, true, true, true, true, true, false, true, true},
        new boolean[] {false, true, false, true, false, true, true, true, false},
        new boolean[] {false, false, false, false, true, true, false, false, true},
        new boolean[] {true, true, true, true, false, true, true, false, false},
        new boolean[] {false, false, true, true, false, true, false, false, true},
        new boolean[] {true, true, false, false, false, true, true, false, true},
        new boolean[] {false, true, true, false, false, false, false, false, false},
        new boolean[] {true, false, false, false, true, true, false, false, true}
      });
  param0.add(
      new boolean[][] {
        new boolean[] {false, false, false, true},
        new boolean[] {false, false, true, true},
        new boolean[] {false, false, false, true},
        new boolean[] {false, false, true, true}
      });
  param0.add(
      new boolean[][] {
        new boolean[] {
          false, true, true, false, false, true, false, false, true, false, false, false, true,
          false, false, true, false, true, true, false, false, true, true, true, true, true, true,
          false, false, true, false, false, false, true, false, true, true, true, false
        },
        new boolean[] {
          false, true, true, true, false, false, true, false, true, true, true, true, true, false,
          true, false, false, true, false, true, true, true, true, false, false, true, false,
          true, true, false, true, false, true, false, false, false, true, true, false
        },
        new boolean[] {
          false, true, true, false, true, false, false, true, false, true, true, false, true,
          true, false, true, false, true, false, true, false, true, true, false, true, false,
          true, false, true, false, false, false, false, true, false, false, true, true, true
        },
        new boolean[] {
          false, false, false, true, false, true, true, true, true, false, true, true, true,
          false, false, false, false, true, false, false, false, true, true, false, true, true,
          false, true, false, false, false, false, false, true, true, true, false, false, false
        },
        new boolean[] {
          true, false, false, true, false, false, false, true, false, true, true, false, false,
          true, true, true, false, false, false, false, false, true, true, false, true, false,
          false, true, false, true, false, false, false, true, false, true, false, true, false
        },
        new boolean[] {
          false, true, true, false, false, true, false, true, false, true, false, true, true,
          true, true, true, false, true, false, false, false, true, true, true, false, false,
          false, false, true, true, true, false, true, false, true, true, false, true, true
        },
        new boolean[] {
          false, false, true, false, true, true, true, true, false, true, true, true, true, false,
          false, true, false, true, false, false, false, true, true, true, false, true, true,
          true, false, false, false, false, false, true, true, false, true, true, false
        },
        new boolean[] {
          false, true, false, false, true, false, false, false, true, false, true, true, true,
          false, true, true, false, false, false, true, true, true, false, true, false, false,
          true, false, true, false, false, true, false, true, true, false, true, false, true
        },
        new boolean[] {
          true, true, true, false, true, true, true, false, false, false, false, true, true,
          false, false, false, true, false, false, true, false, false, false, true, true, false,
          false, false, true, true, false, true, false, true, false, false, false, true, false
        },
        new boolean[] {
          false, false, true, false, true, true, true, false, true, false, false, false, true,
          false, true, false, true, false, false, false, false, true, false, false, true, false,
          true, false, false, true, false, true, true, false, true, false, false, false, false
        },
        new boolean[] {
          true, false, true, true, true, false, true, true, false, true, false, true, false,
          false, false, true, true, true, true, true, false, true, true, false, true, true, true,
          true, false, false, true, false, false, false, false, true, false, false, false
        },
        new boolean[] {
          false, true, true, false, true, false, true, true, true, true, false, false, false,
          false, true, false, true, true, true, false, true, false, false, true, true, true, true,
          false, false, true, false, false, true, false, false, true, false, true, true
        },
        new boolean[] {
          false, false, false, false, true, false, false, true, true, true, false, true, true,
          false, true, false, false, false, true, true, true, true, true, false, false, true,
          false, false, true, false, true, false, false, false, true, true, true, false, false
        },
        new boolean[] {
          false, true, false, true, false, true, true, true, false, false, true, true, true,
          false, false, true, true, false, true, true, false, true, false, true, true, false,
          false, true, false, false, true, false, false, true, true, false, false, false, true
        },
        new boolean[] {
          false, false, true, false, true, true, false, false, false, true, true, true, true,
          true, false, true, false, false, false, false, false, false, true, false, false, false,
          false, false, true, true, false, false, false, true, false, true, true, false, false
        },
        new boolean[] {
          false, true, false, true, true, true, true, false, false, false, true, true, false,
          true, true, false, false, true, false, true, true, true, true, true, false, true, false,
          true, true, true, false, false, true, true, false, false, false, false, false
        },
        new boolean[] {
          true, true, false, false, true, true, true, false, false, false, true, true, true, true,
          false, true, false, false, true, true, false, true, true, true, false, true, true,
          false, false, false, true, true, false, false, false, false, true, false, true
        },
        new boolean[] {
          false, false, false, true, false, false, true, false, true, true, false, true, true,
          true, false, true, false, false, true, true, false, false, true, false, false, true,
          false, false, false, true, false, false, false, true, false, false, false, false, false
        },
        new boolean[] {
          false, true, false, false, true, false, true, true, true, false, true, true, true, true,
          true, false, false, false, true, false, true, true, true, false, true, false, true,
          false, false, true, true, true, true, true, false, true, true, true, true
        },
        new boolean[] {
          true, false, true, false, true, true, false, false, false, true, true, false, true,
          true, true, true, true, false, false, true, false, true, false, true, true, true, true,
          true, false, false, true, true, false, true, false, true, false, false, false
        },
        new boolean[] {
          true, true, false, false, false, false, false, true, true, true, false, true, false,
          true, true, true, false, true, false, true, true, false, true, true, true, false, false,
          true, true, true, false, true, false, true, true, false, true, false, true
        },
        new boolean[] {
          false, false, false, false, true, true, true, false, false, true, true, true, false,
          false, true, true, true, false, true, false, false, true, false, false, true, false,
          true, true, true, true, false, true, true, false, false, true, false, true, true
        },
        new boolean[] {
          false, true, true, false, true, true, true, true, false, false, true, false, false,
          true, true, true, false, false, false, true, true, true, false, true, true, true, true,
          false, true, false, true, false, false, false, true, false, false, true, true
        },
        new boolean[] {
          true, false, false, false, false, true, true, false, false, true, false, false, true,
          true, false, false, true, true, true, false, true, true, false, false, true, false,
          true, false, false, true, true, true, true, true, false, false, true, true, true
        },
        new boolean[] {
          true, true, true, false, false, true, false, true, false, true, true, true, true, false,
          false, true, true, true, false, false, false, true, false, false, false, false, false,
          true, true, true, false, true, true, false, false, false, true, true, true
        },
        new boolean[] {
          true, false, true, true, true, false, false, true, true, false, false, false, true,
          true, false, true, false, true, true, true, false, false, false, true, false, false,
          true, true, true, false, true, false, false, true, true, true, false, false, true
        },
        new boolean[] {
          false, false, false, true, true, false, false, false, true, true, false, false, false,
          true, false, true, false, false, false, false, true, true, true, true, true, true, true,
          false, false, false, false, false, false, false, false, true, false, false, true
        },
        new boolean[] {
          false, false, false, true, false, false, false, true, false, false, true, false, false,
          true, false, true, true, false, true, true, true, true, true, true, false, false, false,
          true, true, true, true, false, false, false, false, false, true, true, true
        },
        new boolean[] {
          false, true, false, true, true, false, true, true, true, true, true, true, false, false,
          true, true, true, true, false, false, true, false, true, false, true, true, true, true,
          true, true, false, true, true, true, true, false, true, true, false
        },
        new boolean[] {
          true, false, false, true, false, true, true, true, true, false, false, true, false,
          false, false, true, true, true, false, false, true, false, false, false, false, true,
          false, true, true, false, false, true, false, false, true, true, true, true, true
        },
        new boolean[] {
          false, true, true, true, false, false, true, false, false, true, false, false, true,
          true, true, false, false, true, false, false, false, true, false, true, true, true,
          false, true, false, false, true, true, false, false, false, true, false, true, false
        },
        new boolean[] {
          false, false, true, false, true, false, false, false, false, true, false, false, false,
          true, true, false, false, true, false, false, true, false, true, false, true, false,
          false, false, true, true, false, true, false, false, false, true, false, true, true
        },
        new boolean[] {
          false, true, false, false, true, true, true, true, true, true, false, false, true,
          false, true, false, false, true, true, true, true, false, false, true, false, true,
          false, true, true, true, true, true, true, false, true, false, false, true, true
        },
        new boolean[] {
          false, false, false, true, true, true, false, false, false, false, true, true, false,
          true, false, false, true, false, false, false, true, true, true, true, false, true,
          false, true, true, true, false, true, true, true, false, false, false, false, false
        },
        new boolean[] {
          false, false, true, true, true, false, true, false, true, true, true, true, false, true,
          false, true, false, false, true, false, false, true, false, true, false, true, false,
          true, true, false, false, false, true, false, false, false, true, false, true
        },
        new boolean[] {
          false, false, false, false, true, true, false, true, false, true, false, true, true,
          true, false, false, false, true, false, false, true, false, false, false, false, false,
          true, false, true, true, true, false, false, true, true, true, true, true, false
        },
        new boolean[] {
          true, true, true, true, false, false, false, true, false, false, false, true, false,
          false, true, false, false, false, false, false, true, true, false, false, false, false,
          false, true, true, true, true, true, true, true, true, false, true, true, true
        },
        new boolean[] {
          true, false, false, true, true, false, true, false, false, false, true, false, true,
          false, false, false, false, true, true, false, false, false, true, false, false, true,
          true, true, false, true, true, false, false, false, false, true, false, false, false
        },
        new boolean[] {
          true, true, false, true, true, false, true, true, false, false, true, true, true, false,
          true, false, true, false, true, false, true, false, true, true, true, true, false,
          false, false, false, false, true, true, false, false, true, true, false, false
        }
      });
  param0.add(
      new boolean[][] {
        new boolean[] {false, false, false, false, false, true, true, true, true},
        new boolean[] {false, false, false, false, true, true, true, true, true},
        new boolean[] {false, false, false, false, false, true, true, true, true},
        new boolean[] {false, false, false, false, false, true, true, true, true},
        new boolean[] {false, false, false, false, false, false, true, true, true},
        new boolean[] {true, true, true, true, true, true, true, true, true},
        new boolean[] {false, false, true, true, true, true, true, true, true},
        new boolean[] {false, false, false, false, false, true, true, true, true},
        new boolean[] {false, false, false, false, false, false, true, true, true}
      });
  param0.add(
      new boolean[][] {
        new boolean[] {
          false, true, true, true, true, false, false, true, false, false, false, true, true,
          false, true, false, false, false, false, true, true, true, true, false, false
        },
        new boolean[] {
          false, true, false, false, false, false, true, true, true, true, false, true, true,
          false, true, true, true, false, true, false, true, true, false, false, true
        },
        new boolean[] {
          true, false, false, false, true, false, false, true, true, false, true, false, true,
          true, false, false, true, false, true, true, true, false, false, true, true
        },
        new boolean[] {
          false, true, true, false, true, true, true, true, false, true, false, false, false,
          true, false, false, false, false, true, false, true, true, false, true, false
        },
        new boolean[] {
          true, true, true, false, true, true, false, false, true, true, false, false, false,
          true, true, false, true, false, false, true, true, false, false, true, false
        },
        new boolean[] {
          true, false, false, true, false, false, true, false, true, true, true, false, false,
          true, false, true, true, false, false, false, false, false, true, true, false
        },
        new boolean[] {
          true, false, false, false, false, false, false, false, true, false, true, false, true,
          false, false, false, true, true, true, true, false, true, false, false, false
        },
        new boolean[] {
          true, false, true, false, false, false, false, false, true, false, true, false, true,
          false, true, false, false, true, true, false, false, true, false, true, false
        },
        new boolean[] {
          true, true, true, false, true, true, true, false, false, false, true, true, false, true,
          true, false, true, true, false, false, false, true, false, true, false
        },
        new boolean[] {
          true, false, true, false, true, false, true, true, false, true, true, false, false,
          true, true, true, true, true, false, false, true, false, true, true, false
        },
        new boolean[] {
          true, false, false, false, true, true, false, false, true, false, false, false, true,
          true, false, true, false, false, true, true, false, false, false, false, true
        },
        new boolean[] {
          false, true, false, true, true, false, true, false, false, true, false, false, false,
          false, false, true, false, true, true, true, false, true, true, false, false
        },
        new boolean[] {
          true, false, false, true, true, false, false, true, false, true, false, false, false,
          true, false, false, true, true, false, true, true, true, true, true, false
        },
        new boolean[] {
          false, true, true, true, true, false, false, false, false, true, true, true, true,
          false, true, true, false, false, true, true, true, true, true, true, false
        },
        new boolean[] {
          true, false, true, false, false, true, false, true, true, true, true, false, true, true,
          false, true, false, true, true, false, true, true, true, false, true
        },
        new boolean[] {
          true, true, true, false, false, false, true, false, true, false, true, false, true,
          true, false, false, true, true, true, false, false, true, true, false, true
        },
        new boolean[] {
          false, false, true, true, true, false, false, false, true, true, false, true, true,
          true, false, true, false, true, true, false, false, false, false, false, false
        },
        new boolean[] {
          false, false, false, true, true, true, true, false, false, true, true, true, false,
          true, true, false, true, true, true, false, false, true, false, true, false
        },
        new boolean[] {
          false, false, true, false, false, true, false, true, false, false, false, false, true,
          false, false, false, false, true, false, true, false, false, true, false, false
        },
        new boolean[] {
          false, false, true, true, false, false, false, true, true, true, false, false, true,
          false, false, true, true, false, false, false, false, true, false, true, false
        },
        new boolean[] {
          true, false, false, false, false, true, false, true, false, false, false, false, true,
          false, true, false, false, true, true, true, false, false, false, true, true
        },
        new boolean[] {
          false, true, false, false, true, false, false, true, false, true, true, true, true,
          false, true, false, true, true, false, true, true, false, false, false, false
        },
        new boolean[] {
          true, false, true, true, false, true, true, true, true, true, true, false, false, true,
          true, true, false, false, false, true, false, true, true, false, false
        },
        new boolean[] {
          true, true, true, false, true, false, true, true, true, false, true, true, true, false,
          false, false, false, true, false, true, true, true, true, false, true
        },
        new boolean[] {
          true, true, true, true, false, true, false, false, false, true, false, false, true,
          false, true, false, true, true, false, false, false, true, false, false, true
        }
      });
  param0.add(
      new boolean[][] {
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, true, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, true, true, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, true, true, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, false, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, true, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, false, false, false, false, true, true, true, true, true, true,
          true, true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, false, false, false, false, false, true, true, true, true, true,
          true, true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, true, true,
          true, true, true, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          true, true, true, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, false, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, false, false, false, true, true, true, true, true, true, true,
          true, true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, true, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, true, true, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, false, false, false, true, true, true, true, true, true, true,
          true, true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, true, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, true, true,
          true, true, true, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, false, false, false, true, true, true, true, true, true, true,
          true, true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, true, true, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, false, false, false, true, true, true, true, true, true, true,
          true, true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, true, true, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, true, true,
          true, true, true, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, false, false, false, false, false, false, false, true, true, true,
          true, true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, true, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, true, true, true, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, false, false, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, false, true, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, false, false, true, true, true, true, true, true, true, true, true,
          true, true, true, true
        },
        new boolean[] {
          false, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, false, false, false, false, true, true, true, true, true, true,
          true, true, true, true, true
        }
      });
  param0.add(
      new boolean[][] {
        new boolean[] {
          true, false, true, false, false, false, true, true, false, true, false, true, true,
          true, false, false, false, false, true, false, true, false, false, true, false, false,
          true
        },
        new boolean[] {
          false, true, false, true, false, false, true, true, false, false, false, true, true,
          false, true, false, true, false, true, false, true, false, false, false, false, true,
          true
        },
        new boolean[] {
          true, false, true, true, true, false, false, true, true, true, true, true, true, true,
          true, false, false, false, true, false, true, true, false, false, true, false, true
        },
        new boolean[] {
          true, true, false, false, false, false, false, false, false, false, false, false, false,
          false, false, false, true, false, true, false, false, true, true, true, false, false,
          true
        },
        new boolean[] {
          true, false, true, false, false, true, true, false, false, true, true, true, false,
          false, false, true, false, true, false, true, true, true, true, true, true, false, false
        },
        new boolean[] {
          true, false, false, true, true, false, false, true, false, true, true, true, true,
          false, false, true, true, true, false, false, true, false, false, true, true, true,
          false
        },
        new boolean[] {
          true, false, true, true, true, true, false, true, false, false, false, true, true,
          false, true, true, false, true, true, false, true, false, false, false, false, false,
          false
        },
        new boolean[] {
          true, false, false, true, false, false, true, true, true, false, true, false, false,
          false, true, true, true, true, false, true, false, false, true, false, true, false, true
        },
        new boolean[] {
          false, true, false, true, false, true, true, true, false, true, true, false, false,
          false, true, false, false, false, false, true, true, false, true, false, true, false,
          true
        },
        new boolean[] {
          false, true, true, true, false, true, false, true, false, true, true, false, true, true,
          true, true, true, true, true, false, true, true, false, false, true, false, true
        },
        new boolean[] {
          false, true, false, false, true, true, false, false, false, false, true, true, false,
          false, true, false, false, true, true, true, true, false, false, true, false, false,
          true
        },
        new boolean[] {
          false, false, true, true, false, true, true, true, false, false, false, false, true,
          false, true, false, true, false, false, true, false, false, true, true, true, false,
          false
        },
        new boolean[] {
          true, true, false, false, true, true, false, false, true, true, true, false, false,
          true, true, false, false, false, true, false, false, false, true, false, false, false,
          true
        },
        new boolean[] {
          false, true, true, true, false, true, true, true, false, false, false, false, false,
          true, true, false, false, false, false, false, true, false, true, true, false, true,
          false
        },
        new boolean[] {
          true, true, true, true, true, true, true, true, true, false, true, true, true, true,
          false, false, false, false, false, true, false, false, false, false, false, false, true
        },
        new boolean[] {
          false, false, false, true, false, false, false, false, false, true, false, false, false,
          false, false, false, false, true, false, true, false, true, false, true, false, true,
          false
        },
        new boolean[] {
          true, true, false, true, true, true, true, true, true, false, false, true, true, false,
          true, true, false, false, false, false, false, true, true, false, false, false, false
        },
        new boolean[] {
          false, false, false, false, true, true, true, false, true, true, false, true, false,
          false, true, true, false, false, false, false, true, true, false, true, true, false,
          false
        },
        new boolean[] {
          true, false, true, true, false, true, false, false, false, false, false, false, false,
          false, true, false, true, true, false, true, true, true, true, false, false, false, true
        },
        new boolean[] {
          true, false, false, false, true, false, true, false, true, true, false, false, false,
          true, false, true, true, true, false, false, false, true, false, true, true, false, true
        },
        new boolean[] {
          true, false, true, true, true, true, false, true, true, false, true, true, true, false,
          false, true, true, false, false, false, false, false, true, false, true, true, true
        },
        new boolean[] {
          true, true, false, false, false, true, false, true, true, true, true, false, true, true,
          true, true, true, true, false, false, false, false, true, true, false, false, false
        },
        new boolean[] {
          true, false, false, false, false, false, false, true, true, true, false, true, false,
          false, false, false, true, false, false, false, true, true, false, true, true, true,
          true
        },
        new boolean[] {
          false, true, true, true, true, false, false, false, true, true, false, true, false,
          false, false, true, false, false, true, true, true, false, false, false, true, true,
          true
        },
        new boolean[] {
          false, false, true, true, false, true, true, false, false, true, true, true, false,
          false, true, false, true, true, true, true, false, true, true, true, true, false, false
        },
        new boolean[] {
          true, true, false, true, false, true, false, true, true, false, false, true, false,
          false, true, true, false, false, true, true, false, true, true, true, true, false, false
        },
        new boolean[] {
          true, false, true, false, true, true, true, true, true, false, false, false, false,
          false, false, true, true, false, false, false, false, false, false, true, false, true,
          true
        }
      });
  param0.add(
      new boolean[][] {
        new boolean[] {false, false, false, true},
        new boolean[] {false, true, true, true},
        new boolean[] {false, false, false, true},
        new boolean[] {false, true, true, true}
      });
  param0.add(
      new boolean[][] {
        new boolean[] {
          true, true, false, false, true, true, true, true, true, false, true, true, false, true,
          true, false, false, false, false, false, true, false, true, false, true, true, false,
          true
        },
        new boolean[] {
          false, false, true, true, false, false, false, true, true, false, false, true, false,
          true, false, false, true, true, false, false, true, true, true, true, false, true,
          false, false
        },
        new boolean[] {
          true, true, false, false, false, true, false, true, true, true, false, true, false,
          true, false, false, true, true, false, true, true, false, true, true, false, false,
          false, false
        },
        new boolean[] {
          true, false, true, false, true, false, true, false, false, true, true, true, true, true,
          true, false, true, false, false, true, false, false, false, true, false, true, false,
          true
        },
        new boolean[] {
          true, true, true, true, false, false, false, true, true, false, true, false, true,
          false, true, true, true, true, false, false, true, true, true, true, false, true, true,
          true
        },
        new boolean[] {
          true, false, true, true, true, true, true, true, false, true, false, false, false,
          false, false, true, false, true, true, false, true, true, false, true, false, false,
          false, true
        },
        new boolean[] {
          true, true, false, false, false, true, true, false, true, false, true, false, false,
          false, true, true, true, false, false, true, true, false, true, false, false, false,
          true, false
        },
        new boolean[] {
          false, true, true, true, true, false, false, true, false, false, false, false, false,
          false, false, false, true, false, true, false, false, true, false, true, true, true,
          false, true
        },
        new boolean[] {
          true, false, true, false, false, false, true, false, true, true, true, true, false,
          true, true, true, false, false, true, true, false, false, false, false, true, false,
          false, false
        },
        new boolean[] {
          false, false, true, true, false, true, false, false, true, true, true, true, false,
          false, true, false, false, true, true, false, true, false, true, true, false, true,
          true, true
        },
        new boolean[] {
          true, false, true, true, true, true, false, true, true, true, false, true, true, false,
          false, false, true, false, true, true, true, true, true, false, false, false, false,
          false
        },
        new boolean[] {
          false, false, false, false, true, false, true, true, true, false, false, false, false,
          true, false, false, true, true, false, true, true, true, true, true, true, true, true,
          false
        },
        new boolean[] {
          false, false, false, true, true, false, false, true, false, false, false, false, true,
          true, true, true, false, false, true, true, true, true, true, true, true, true, false,
          false
        },
        new boolean[] {
          false, true, true, true, true, true, true, true, true, true, true, true, true, true,
          false, true, true, true, true, true, false, false, false, false, true, false, true,
          false
        },
        new boolean[] {
          false, true, false, false, false, true, true, false, false, true, false, true, false,
          true, false, true, true, false, true, true, false, false, true, false, true, false,
          false, true
        },
        new boolean[] {
          true, true, false, true, true, true, true, true, false, false, false, true, true, false,
          false, true, true, true, false, false, false, false, true, false, true, true, false,
          true
        },
        new boolean[] {
          true, false, true, false, false, false, true, true, false, true, true, false, true,
          true, true, true, true, true, true, false, false, false, false, false, false, false,
          false, true
        },
        new boolean[] {
          true, false, true, false, true, false, false, false, true, true, true, false, true,
          true, true, false, false, false, false, false, true, true, true, true, true, true,
          false, false
        },
        new boolean[] {
          true, false, true, false, true, true, true, false, false, false, false, false, false,
          false, true, true, false, false, false, true, true, true, true, false, true, false,
          false, false
        },
        new boolean[] {
          false, false, true, false, true, false, true, false, true, true, false, true, true,
          true, false, false, true, true, true, false, false, false, false, false, false, false,
          false, false
        },
        new boolean[] {
          true, false, true, false, true, true, true, true, false, true, true, false, false, true,
          true, false, true, false, true, true, true, true, true, true, false, false, true, false
        },
        new boolean[] {
          true, false, false, true, false, false, false, false, false, true, true, false, false,
          true, false, false, true, false, true, false, true, false, true, true, false, true,
          false, false
        },
        new boolean[] {
          false, true, true, true, true, true, true, false, false, true, true, false, true, false,
          true, true, true, false, true, true, true, true, false, true, false, false, false, false
        },
        new boolean[] {
          true, true, false, false, true, true, false, false, true, false, false, false, true,
          false, false, false, false, false, true, true, true, false, true, true, false, false,
          true, false
        },
        new boolean[] {
          false, true, true, true, true, true, true, true, false, true, false, false, false, true,
          true, false, false, true, true, false, false, true, false, true, true, false, true,
          false
        },
        new boolean[] {
          true, true, true, true, true, true, true, true, false, true, false, false, true, false,
          true, false, true, true, true, true, false, false, true, false, true, false, true, true
        },
        new boolean[] {
          true, false, true, true, true, false, false, true, false, true, true, false, false,
          false, true, true, true, false, false, true, false, false, true, true, true, true,
          false, true
        },
        new boolean[] {
          false, true, true, false, false, false, true, true, true, true, false, true, true,
          false, false, false, true, true, true, true, false, true, true, true, true, false, true,
          false
        }
      });
  List<Integer> param1 = new ArrayList<>();
  param1.add(2);
  param1.add(4);
  param1.add(2);
  param1.add(30);
  param1.add(7);
  param1.add(13);
  param1.add(19);
  param1.add(15);
  param1.add(3);
  param1.add(18);
  for (int i = 0; i < param0.size(); ++i) {
    if (f_filled(param0.get(i), param1.get(i)) == f_gold(param0.get(i), param1.get(i))) {
      n_success += 1;
    }
  }
  System.out.println("#Results:" + n_success + ", " + param0.size());
}}