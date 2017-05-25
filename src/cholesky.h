//-*-c++-*-
#ifndef CHOLESKY_H
#define CHOLESKY_H

#include <numeric>
#include <cstdio>
#include <cmath>
#include <immintrin.h>

#include "util.h"

//#define L(i, j) L[i * N + j]
#define L(i, j) L[((i * (i + 1))>>1) - 1 + j]
#define _mm256_load_pd _mm256_loadu_pd

/*
// for float
#define REDUCE_ADD(target){\
tmp1 = _mm256_permute_ps(target, 0b10110001);\
tmp1 = _mm256_add_ps(target, tmp1);\
tmp2 = _mm256_permute_ps(tmp1, 0b01001110);\
tmp2 = _mm256_add_ps(tmp2, tmp1);\
tmp3 = _mm256_permute2f128_ps(tmp2, tmp2, 0b00000001);\
target = _mm256_add_ps(tmp3, tmp2);\
}
*/

// for double
#define REDUCE_ADD(target){\
tmp1 = _mm256_permute_pd(target, 0b0101);\
tmp1 = _mm256_add_pd(target, tmp1);\
tmp2 = _mm256_permute2f128_pd(tmp1, tmp1, 0b00000001);\
target = _mm256_add_pd(tmp1, tmp2);\
}

const Real EPSILON = 1e-9;
const int VEC_SIZE = 4;

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

inline void update_cholesky_n_solve(Real *L, Real *w, const int n, const int N,
                  const Real *Xt, const int cur, const Idx *beta, const int D) {
  __m256d sum_v1, sum_v2, sum_v3, sum_v4, sum_v5, sum_v6, sum_v7, sum_v8;
  __m256d tmp0, tmp1, tmp2, tmp3; //for macros
  Real tmp_arr[2 * VEC_SIZE];
  Real sum_s1, sum_s2;

  int i, k;

  for (i = 0; i <= n; ++i) {
    L(n, i) = 0.0;
    for (int k = 0; k < D; k++) {
      L(n, i) += Xt[cur * D + k] * Xt[beta[i].id * D + k];
    }
  }
  
  /* solve (L^-1)X'v and (L^-1)w with Gaussian elimination */
  for (i = 0; i < n; ++i) {
    sum_v1 = _mm256_setzero_pd();
    sum_v2 = _mm256_setzero_pd();
    sum_v3 = _mm256_setzero_pd();
    sum_v4 = _mm256_setzero_pd();
    sum_v5 = _mm256_setzero_pd();
    sum_v6 = _mm256_setzero_pd();
    sum_v7 = _mm256_setzero_pd();
    sum_v8 = _mm256_setzero_pd();
    for (k = 0; k <= i - 16; k += 16) {
      Real *L_i_k = &L(i, k);
      Real *L_n_k = &L(n, k);
      Real *w_k = w + k;

      __m256d vec_L_i_k = _mm256_load_pd(L_i_k);
      __m256d vec_L_i_k_4 = _mm256_load_pd(L_i_k + 4);
      __m256d vec_L_i_k_8 = _mm256_load_pd(L_i_k + 8);
      __m256d vec_L_i_k_12 = _mm256_load_pd(L_i_k + 12);

      sum_v1 = _mm256_fmadd_pd(vec_L_i_k, _mm256_load_pd(L_n_k), sum_v1);
      sum_v2 = _mm256_fmadd_pd(vec_L_i_k_4, _mm256_load_pd(L_n_k + 4), sum_v2);
      sum_v3 = _mm256_fmadd_pd(vec_L_i_k_8, _mm256_load_pd(L_n_k + 8), sum_v3);
      sum_v4 = _mm256_fmadd_pd(vec_L_i_k_12, _mm256_load_pd(L_n_k + 12), sum_v4);

      sum_v5 = _mm256_fmadd_pd(vec_L_i_k, _mm256_load_pd(w_k), sum_v5);
      sum_v6 = _mm256_fmadd_pd(vec_L_i_k_4, _mm256_load_pd(w_k + 4), sum_v6);
      sum_v7 = _mm256_fmadd_pd(vec_L_i_k_8, _mm256_load_pd(w_k + 8), sum_v7);
      sum_v8 = _mm256_fmadd_pd(vec_L_i_k_12, _mm256_load_pd(w_k + 12), sum_v8);
    }
    sum_v1 = _mm256_add_pd(_mm256_add_pd(sum_v1, sum_v3), _mm256_add_pd(sum_v2, sum_v4));
    sum_v5 = _mm256_add_pd(_mm256_add_pd(sum_v5, sum_v7), _mm256_add_pd(sum_v6, sum_v8));
    REDUCE_ADD(sum_v1)
    REDUCE_ADD(sum_v5)
    _mm256_store_pd(tmp_arr, sum_v1);
    _mm256_store_pd(tmp_arr + VEC_SIZE, sum_v5);
    sum_s1 = tmp_arr[0];
    sum_s2 = tmp_arr[VEC_SIZE];
    for (; k < i; ++k) {
      sum_s1 += L(i, k) * L(n, k);
      sum_s2 += L(i, k) * w[k];
    }
    L(n, i) = (L(n, i) - sum_s1) / L(i, i);
    w[i] = (w[i] - sum_s2) / L(i, i);
  }
  /* compute the lower right entry of L and solve the last element of (L^-1)w */
  sum_v1 = _mm256_setzero_pd();
  sum_v2 = _mm256_setzero_pd();
  sum_v3 = _mm256_setzero_pd();
  sum_v4 = _mm256_setzero_pd();
  sum_v5 = _mm256_setzero_pd();
  sum_v6 = _mm256_setzero_pd();
  sum_v7 = _mm256_setzero_pd();
  sum_v8 = _mm256_setzero_pd();
  for (k = 0; k <= i - 16; k += 16) {
    Real *L_n_k = &L(n, k);
    Real *w_k = w + k;

    __m256d vec_L_n_k = _mm256_load_pd(L_n_k);
    __m256d vec_L_n_k_4 = _mm256_load_pd(L_n_k + 4);
    __m256d vec_L_n_k_8 = _mm256_load_pd(L_n_k + 8);
    __m256d vec_L_n_k_12 = _mm256_load_pd(L_n_k + 12);

    sum_v1 = _mm256_fmadd_pd(vec_L_n_k, vec_L_n_k, sum_v1);
    sum_v2 = _mm256_fmadd_pd(vec_L_n_k_4, vec_L_n_k_4, sum_v2);
    sum_v3 = _mm256_fmadd_pd(vec_L_n_k_8, vec_L_n_k_8, sum_v3);
    sum_v4 = _mm256_fmadd_pd(vec_L_n_k_12, vec_L_n_k_12, sum_v4);

    sum_v5 = _mm256_fmadd_pd(vec_L_n_k, _mm256_load_pd(w_k), sum_v5);
    sum_v6 = _mm256_fmadd_pd(vec_L_n_k_4, _mm256_load_pd(w_k + 4), sum_v6);
    sum_v7 = _mm256_fmadd_pd(vec_L_n_k_8, _mm256_load_pd(w_k + 8), sum_v7);
    sum_v8 = _mm256_fmadd_pd(vec_L_n_k_12, _mm256_load_pd(w_k + 12), sum_v8);
  }
  sum_v1 = _mm256_add_pd(_mm256_add_pd(sum_v1, sum_v3), _mm256_add_pd(sum_v2, sum_v4));
  sum_v5 = _mm256_add_pd(_mm256_add_pd(sum_v5, sum_v7), _mm256_add_pd(sum_v6, sum_v8));
  REDUCE_ADD(sum_v1)
  REDUCE_ADD(sum_v5)
  _mm256_store_pd(tmp_arr, sum_v1);
  _mm256_store_pd(tmp_arr + VEC_SIZE, sum_v5);
  sum_s1 = tmp_arr[0];
  sum_s2 = tmp_arr[VEC_SIZE];
  for (; k < n; k++) {
    sum_s1 += L(n, k) * L(n, k);
    sum_s2 += L(n, k) * w[k];
  }
  L(n, n) = sqrt(fmax(L(n, n) - sum_s1, EPSILON));
  w[n] = (w[n] - sum_s2) / L(n, n);

  /* solve ((L')^-1)w with Gaussian elimination */
  for (i = n; i>= 0; i--) {
    w[i] /= L(i, i);
    for (k = 0; k < i; k++) {
      w[k] -= L(i, k) * w[i];
    }
  }
}
#endif
