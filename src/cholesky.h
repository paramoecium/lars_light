//-*-c++-*-
#ifndef CHOLESKY_H
#define CHOLESKY_H

#include <numeric>
#include <cstdio>
#include <cmath>

#include "util.h"

const Real EPSILON = 1e-9;

/////////////
// Methods //
/////////////

// Updates the cholesky (L) after having added data to row (j)
// update L of the gram matrix after including new vector j
// X = LL', cholesky decomposition
// allocated memory of L is N x N x sizeof(Real)
inline void update_cholesky(Real* L, int j, const int N) {
  Real sum = 0.0;
  Real eps_small = EPSILON;
  int i, k;
  for (i = 0; i < j; ++i) {
    sum = L[j * N + i];
    for (k = 0; k < i; ++k) {
      sum -= L[i * N + k] * L[j * N + k];
    }
    L[j * N + i] = sum / L[i * N + i];
  }
  sum = L[j * N + i];
  for (k = 0; k < i; k++) {
    sum -= L[i * N + k] * L[j * N + k];
  }
  if (sum <= 0.0) sum = eps_small;
  L[j * N + j] = sqrt(sum);
}

// Backsolve the cholesky (L) for unknown (x) given a right-hand-side (b)
// and a total number of rows/columns (n).
//
// Solves for x in Lx=b
// x can be b
// => Assume L is nxn, x and b is vector of length n
// assume L = nxn matrix
// current L is of nxn size, but the memory is stored in a NxN data structure
inline void backsolve(const Real *L, Real *x, const Real *b, const int n, const int N) {
  int i, k;
  Real sum;
  for (i = 0; i < n; i++) {
    for (sum = b[i], k = i-1; k >= 0; k--) {
      sum -= L[i * N + k] * x[k];
    }
    x[i] = sum / L[i * N + i];
  }

  for (i = n-1; i>= 0; i--) {
    for (sum = x[i], k = i+1; k < n; k++) {
      sum -= L[k * N + i] * x[k];
    }
    x[i] = sum / L[i * N + i];
  }
}

// Downdates the cholesky (L) by removing row (id), given that there
// are (nrows) total rows.
//
// Given that A = L * L';
// We correct L if row/column id is removed from A.
// We do not resize L in this function, but we set its last row to 0, since
// L is one row/col smaller after this function
// n is the number of rows and cols in the cholesky
// void downdate_cholesky(Real *L, int nrows, int id) {
//   Real a = 0, b = 0, c = 0, s = 0, tau = 0;
//   int lth = nrows - 1;
//   int L_rows = nrows, L_cols = nrows;
//
//   int i, j;
//   for (i = id; i < lth; ++i) {
//     for (j = 0; j < i; ++j) {
//       L[i * L_cols + j] = L[(i+1) * L_cols + j];
//     }
//     a = L[(i+1) * L_cols + i];
//     b = L[(i+1) * L_cols + (i + 1)];
//     if (b == 0) {
//       L[i * L_cols + i] = a;
//       continue;
//     }
//     if (fabs(b) > fabs(a)) {
//       tau = -a / b;
//       s = 1.0 / sqrt(1.0 + tau*tau);
//       c = s*tau;
//     } else {
//       tau = -b / a;
//       c = 1.0 / sqrt(1.0 + tau*tau);
//       s = c * tau;
//     }
//     L[i * L_cols + i] = c * a - s * b;
//     for (j = i+2; j <= lth; ++j) {
//       a = L[j * L_cols + i];
//       b = L[j * L_cols + (i+1)];
//       L[j * L_cols + i] = c*a - s*b;
//       L[J * L_cols + (i+1)] = s*a + c*b;
//     }
//   }
//   for (i = 0; i <= lth; ++i) {
//     L[lth * L_cols + i] = 0.0;
//   }
// }
#endif
