//-*-c++-*-
#ifndef CHOLESKY_H
#define CHOLESKY_H

#include <numeric>
#include <cstdio>
#include <cmath>

#include "util.h"

const Real EPSILON = 1e-9;

/*
1.
X'X = LL', L is a n x n matrix in N x N memory
Update cholesky decomposition L with Gaussian elimination of the gram matrix X'X
after including new vector
L[n * N : n * N + N] stores the inner product of the new vector and the other vectors
in the active set(including itself)
2.
Compute ((X'X)^-1)w by solving (X'X)w = (LL')w = w
*/

inline void update_cholesky_n_solve(Real *L, Real *w, const int n, const int N) {
  Real sum0, sum1;
  int i, k;
  /* solve (L^-1)X'v and (L^-1)w with Gaussian elimination */
  for (i = 0; i < n; ++i) {
    sum0 = 0.0, sum1 = 0.0;
    for (k = 0; k < i; ++k) {
      sum0 += L[i * N + k] * L[n * N + k];
      sum1 += L[i * N + k] * w[k];
    }
    L[n * N + i] = (L[n * N + i] - sum0) / L[i * N + i];
    w[i] = (w[i] - sum1) / L[i * N + i];
  }
  /* compute the lower right entry of L and solve the last element of (L^-1)w */
  sum0 = 0.0;
  sum1 = 0.0;
  for (k = 0; k < n; k++) {
    sum0 += L[n * N + k] * L[n * N + k];
    sum1 += L[n * N + k] * w[k];
  }
  L[n * N + n] = sqrt(fmax(L[n * N + n] - sum0, EPSILON));
  w[n] = (w[n] - sum1) / L[n * N + n];

  /* solve ((L')^-1)w with Gaussian elimination */
  for (i = n; i>= 0; i--) {
    w[i] /= L[i * N + i];
    for (k = 0; k < i; k++) {
      w[k] -= L[i * N + k] * w[i];
    }
  }
}
#endif
