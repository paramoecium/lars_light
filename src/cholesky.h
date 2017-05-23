//-*-c++-*-
#ifndef CHOLESKY_H
#define CHOLESKY_H

#include <numeric>
#include <cstdio>
#include <cmath>

#include "util.h"

const Real EPSILON = 1e-9;
const int B = 56; //block size;
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
  int i, k, b;
  /* solve L^-1 with Gaussian elimination */
  /*
  for (i = 0; i < j; ++i) {
    sum = 0.0;
    for (k = 0; k < i; ++k) {
      sum += L[i * N + k] * L[j * N + k];
    }
    L[j * N + i] = (L[j * N + i] - sum) / L[i * N + i];
  }
  */
  for (b = 0; b + B <= j; b += B) {
    /* solving the top triangle */
    for (i = b; i < b + B; i++) {
      sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
      for (k = b; k <= i - 4; k += 4) {
        sum1 += L[i * N + k] * L[j * N + k];
        sum2 += L[i * N + k + 1] * L[j * N + k + 1];
        sum3 += L[i * N + k + 2] * L[j * N + k + 2];
        sum4 += L[i * N + k + 3] * L[j * N + k + 3];
      }
      for (; k < i; ++k) {
        sum1 += L[i * N + k] * L[j * N + k];
      }
      L[j * N + i] = (L[j * N + i] - sum1 - sum2 - sum3 - sum4) / L[i * N + i];
    }
    /* solving the rectangle below */
    for (;i < j; i++) {
      sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
      for (k = b; k <= b + B - 4; k += 4) {
        sum1 += L[i * N + k] * L[j * N + k];
        sum2 += L[i * N + k + 1] * L[j * N + k + 1];
        sum3 += L[i * N + k + 2] * L[j * N + k + 2];
        sum4 += L[i * N + k + 3] * L[j * N + k + 3];
      }
      for (; k < b + B; k++) {
        sum1 += L[i * N + k] * L[j * N + k];
      }
      L[j * N + i] -= sum1 + sum2 + sum3 + sum4;
    }
  }
  /* finish the remaining triangle */
  for (i = b; i < j; i++) {
    sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
    for (k = b; k <= i - 4; k += 4) {
      sum1 += L[i * N + k] * L[j * N + k];
      sum2 += L[i * N + k + 1] * L[j * N + k + 1];
      sum3 += L[i * N + k + 2] * L[j * N + k + 2];
      sum4 += L[i * N + k + 3] * L[j * N + k + 3];
    }
    for (; k < i; ++k) {
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
  Real sum;
  /* solve L^-1 with Gaussian elimination */
  for (i = 0; i < n; i++) {
    sum = 0.0;
    for (k = 0; k < i; ++k) {
      sum += L[i * N + k] * w[k];
    }
    w[i] = (v[i] - sum) / L[i * N + i];
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
