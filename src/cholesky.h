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

/*
X'X = LL', L is a n x n matrix in N x N memory
Update cholesky decomposition L of the gram matrix X'X
after including new vector j with Gaussian elimination
L[j * N : j * N + N] stores the inner product of the vector j and all vectors
in the active set(including itself)
*/
inline void update_cholesky(float* L, int j, const int N) {
  float sum = 0.0;
  float eps_small = EPSILON;
  int i, k;
  for (i = 0; i < j; ++i) {
    sum = L[j * N + i];
    for (k = 0; k < i; ++k) {
      sum -= L[i * N + k] * L[j * N + k];
    }
    L[j * N + i] = sum / L[i * N + i];
  }
  sum = L[j * N + j];
  for (k = 0; k < j; k++) {
    sum -= L[j * N + k] * L[j * N + k];
  }
  if (sum <= 0.0) sum = eps_small;
  L[j * N + j] = sqrt(sum);
}
/*
X'X = LL', L is a n x n matrix in N x N memory, w and v are vectors of length n
Solve for w in (X'X)w = (LL')w = v, where w can be v
*/
inline void backsolve(const Real *L, Real *w, const Real *v, const int n, const int N) {
  int i, k;
  Real sum;
  for (i = 0; i < n; i++) {
    sum = v[i];
    for (k = 0; k < i; ++k) {
      sum -= L[i * N + k] * w[k];
    }
    w[i] = sum / L[i * N + i];
  }

  for (i = n-1; i>= 0; i--) {
    sum = w[i];
    for (k = i+1; k < n; k++) {
      sum -= L[k * N + i] * w[k];
    }
    w[i] = sum / L[i * N + i];
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
