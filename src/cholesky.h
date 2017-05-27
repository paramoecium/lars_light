//-*-c++-*-
#ifndef CHOLESKY_H
#define CHOLESKY_H

#include <numeric>
#include <cstdio>
#include <cmath>

#include "util.h"

const Real EPSILON = 1e-9;
const int B = 64; //block size;

//#define L(i, j) L[i * N + j]
#define L(i, j) L[((i * (i + 1))>>1) - 1 + j]

inline void update_gram_matrix(Real *L, const int active_itr, const int N,
                               const Real *Xt, const int cur, const Idx *beta, const int D) {
  for (int i = 0; i <= active_itr; ++i) {
    L(active_itr, i) = dot(Xt + cur * D, Xt + beta[i].id * D, D);
  }
}
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
  int i, k, b_i, b_k;
  /* solve (L^-1)X'v and (L^-1)w with Gaussian elimination */
  for (b_i = 0; b_i + B <= n; b_i += B) {
    /* compute mvms (BxB)(Bx1) */
    for (b_k = 0; b_k + B <= b_i; b_k += B) {
      for (i = b_i; i < b_i + B; i++) {
        sum0 = 0.0, sum1 = 0.0;
        for (k = b_k; k < b_k + B; k++) {
          sum0 += L(i, k) * L(n, k);
          sum1 += L(i, k) * w[k];
        }
        L(n, i) -= sum0;
        w[i] -= sum1;
      }
    }
    /* pivot the triangle */
    for (i = b_i; i < b_i + B; i++) {
      sum0 = 0.0, sum1 = 0.0;
      for (k = b_k; k < i; ++k) {
        sum0 += L(i, k) * L(n, k);
        sum1 += L(i, k) * w[k];
      }
      L(n, i) = (L(n, i) - sum0) / L(i, i);
      w[i] = (w[i] - sum1) / L(i, i);

    }
  }
  /* finish the remaining triangle */
  for (i = b_i; i < n; i++) {
    sum0 = 0.0, sum1 = 0.0;
    for (k = 0; k < i; ++k) {
      sum0 += L(i, k) * L(n, k);
      sum1 += L(i, k) * w[k];
    }
    L(n, i) = (L(n, i) - sum0) / L(i, i);
    w[i] = (w[i] - sum1) / L(i, i);
  }
  /* compute the lower right entry of L and solve the last element of (L^-1)w */
  sum0 = 0.0;
  sum1 = 0.0;
  for (k = 0; k < n; k++) {
    sum0 += L(n, k) * L(n, k);
    sum1 += L(n, k) * w[k];
  }
  L(n, n) = sqrt(fmax(L(n, n) - sum0, EPSILON));
  w[n] = (w[n] - sum1) / L(n, n);

  /* solve ((L')^-1)w with Gaussian elimination */
  for (b_i = n; b_i - B >= -1; b_i -= B) {
    /* pivot the triangle */
    for (i = b_i; i > b_i - B; i--) {
      w[i] /= L(i, i);
      for (k = b_i - B + 1; k < i; k++) {
        w[k] -= L(i, k) * w[i];
      }
    }
    /* compute mvms (BxB)(Bx1) */
    for (b_k = b_i - B; b_k - B >= -1; b_k -= B) {
      for (i = b_i; i > b_i - B; i--) {
        for (k = b_k - B + 1; k <= b_k; k++) {
          w[k] -= L(i, k) * w[i];
        }
      }
    }
    /* finish the remaining rectangle */
    for (i = b_i; i > b_i - B; i--) {
      for (k = 0; k <= b_k; k++) {
        w[k] -= L(i, k) * w[i];
      }
    }
  }
  /* finish the remaining triangle */
  for (i = b_i; i>= 0; i--) {
    w[i] /= L(i, i);
    for (k = 0; k < i; k++) {
      w[k] -= L(i, k) * w[i];
    }
  }
}
#endif
