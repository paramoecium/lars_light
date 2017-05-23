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
inline void update_cholesky(Real* L, int j, const int N) {
  Real sum1, sum2, sum3, sum4;
  int i, k;
  /* solve L^-1 with Gaussian elimination */
  for (i = 0; i < j; ++i) {
    sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
    for (k = 0; k <= i - 4; k += 4) {
      sum1 += L[i * N + k] * L[j * N + k];
      sum2 += L[i * N + k + 1] * L[j * N + k + 1];
      sum3 += L[i * N + k + 2] * L[j * N + k + 2];
      sum4 += L[i * N + k + 3] * L[j * N + k + 3];
    }
    for (; k <= i - 2; k += 2) {
      sum1 += L[i * N + k] * L[j * N + k];
      sum2 += L[i * N + k + 1] * L[j * N + k + 1];
    }
    for (; k < i; k++) {
      sum1 += L[i * N + k] * L[j * N + k];
    }
    L[j * N + i] = (L[j * N + i] - sum1 - sum2 - sum3 - sum4) / L[i * N + i];
  }
  /* compute the lower right entry */
  sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
  for (k = 0; k <= j - 4; k += 4) {
    sum1 += L[j * N + k] * L[j * N + k];
    sum2 += L[j * N + k + 1] * L[j * N + k + 1];
    sum3 += L[j * N + k + 2] * L[j * N + k + 2];
    sum4 += L[j * N + k + 3] * L[j * N + k + 3];
  }
  for (; k <= j - 2; k += 2) {
    sum1 += L[j * N + k] * L[j * N + k];
    sum2 += L[j * N + k + 1] * L[j * N + k + 1];
  }
  for (; k < j; k++) {
    sum1 += L[j * N + k] * L[j * N + k];
  }
  sum1 = L[j * N + j] - sum1 - sum2 - sum3 - sum4;
  if (sum1 <= 0.0) sum1 = EPSILON;
  L[j * N + j] = sqrt(sum1);
}
/*
X'X = LL', L is a n x n matrix in N x N memory, w and v are vectors of length n
Solve for w in (X'X)w = (LL')w = v, where w can be v
*/
inline void backsolve(const Real *L, Real *w, const Real *v,
                      const int n, const int N) {
  int i, k;
  Real sum1, sum2, sum3, sum4;
  /* solve L^-1 with Gaussian elimination */
  for (i = 0; i < n; i++) {
    sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
    for (k = 0; k <= i - 4; k += 4) {
      sum1 += L[i * N + k] * w[k];
      sum2 += L[i * N + k + 1] * w[k + 1];
      sum3 += L[i * N + k + 2] * w[k + 2];
      sum4 += L[i * N + k + 3] * w[k + 3];
    }
    for (; k <= i - 2; k += 2) {
      sum1 += L[i * N + k] * w[k];
      sum2 += L[i * N + k + 1] * w[k + 1];
    }
    for (; k < i; k++) {
      sum1 += L[i * N + k] * w[k];
    }
    w[i] = (v[i] - sum1 - sum2 - sum3 - sum4) / L[i * N + i];
  }
  /* solve (L')^-1 with Gaussian elimination */
  for (i = n-1; i>= 0; i--) {
    w[i] /= L[i * N + i];
    for (k = 0; k < i; k++) {
      w[k] -= L[i * N + k] * w[i];
    }
  }
}
#endif
