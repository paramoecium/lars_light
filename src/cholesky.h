//-*-c++-*-
#ifndef CHOLESKY_H
#define CHOLESKY_H

#include <numeric>
#include <cstdio>
#include <cmath>
#include <immintrin.h>

#include "util.h"

const Real EPSILON = 1e-9;
const int VEC_SIZE = 4;

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

/*
X'X = LL', L is a n x n matrix in N x N memory
Update cholesky decomposition L of the gram matrix X'X
after including new vector j with Gaussian elimination
L[j * N : j * N + N] stores the inner product of the vector j and all vectors
in the active set(including itself)
*/
inline void update_cholesky(Real* L, int j, const int N) {
  __m256d sum1, sum2;
  __m256d tmp0, tmp1, tmp2, tmp3; //for macros
  Real tmp_arr[VEC_SIZE];
  Real sum;
  int i, k;
  /* solve L^-1 with Gaussian elimination */
  for (i = 0; i < j; ++i) {
    sum1 = _mm256_setzero_pd();
    sum2 = _mm256_setzero_pd();
    for (k = 0; k <= i - 2 * VEC_SIZE; k += 2 * VEC_SIZE) {
      Real *L_i_N_k = L + i * N + k;
      Real *L_j_N_k = L + j * N + k;
      sum1 = _mm256_fmadd_pd(_mm256_load_pd(L_i_N_k), _mm256_load_pd(L_j_N_k), sum1);
      sum2 = _mm256_fmadd_pd(_mm256_load_pd(L_i_N_k + VEC_SIZE), _mm256_load_pd(L_j_N_k + VEC_SIZE), sum2);
    }
    sum1 = _mm256_add_pd(sum1, sum2);
    REDUCE_ADD(sum1)
    _mm256_store_pd(tmp_arr, sum1);
    sum = tmp_arr[0];
    for (; k < i; k++) {
      sum += L[i * N + k] * L[j * N + k];
    }
    L[j * N + i] = (L[j * N + i] - sum) / L[i * N + i];
  }
  /* compute the lower right entry */
  sum1 = _mm256_setzero_pd();
  sum2 = _mm256_setzero_pd();
  for (k = 0; k <= j - 2 * VEC_SIZE; k += 2 * VEC_SIZE) {
    __m256d L_i_N_k = _mm256_load_pd(L + j * N + k);
    __m256d L_i_N_k_4 = _mm256_load_pd(L + j * N + k + VEC_SIZE);
    sum1 = _mm256_fmadd_pd(L_i_N_k, L_i_N_k, sum1);
    sum2 = _mm256_fmadd_pd(L_i_N_k_4, L_i_N_k_4, sum2);
  }
  sum1 = _mm256_add_pd(sum1, sum2);
  REDUCE_ADD(sum1)
  _mm256_store_pd(tmp_arr, sum1);
  sum = tmp_arr[0];
  for (; k < j; k++) {
    sum += L[j * N + k] * L[j * N + k];
  }
  sum = L[j * N + j] - sum;
  if (sum <= 0.0) sum = EPSILON;
  L[j * N + j] = sqrt(sum);
}
/*
X'X = LL', L is a n x n matrix in N x N memory, w and v are vectors of length n
Solve for w in (X'X)w = (LL')w = v, where w can be v
*/
inline void backsolve(const Real *L, Real *w, const Real *v,
                      const int n, const int N) {
  __m256d sum1, sum2;
  __m256d tmp0, tmp1, tmp2, tmp3; //for macros
  Real tmp_arr[VEC_SIZE];
  Real sum;
  int i, k;
  /* solve L^-1 with Gaussian elimination */
  for (i = 0; i < n; i++) {
    sum1 = _mm256_setzero_pd();
    sum2 = _mm256_setzero_pd();
    for (k = 0; k <= i - 2 * VEC_SIZE; k += 2 * VEC_SIZE) {
      sum1 = _mm256_fmadd_pd(_mm256_load_pd(L + i * N + k), _mm256_load_pd(w + k), sum1);
      sum2 = _mm256_fmadd_pd(_mm256_load_pd(L + i * N + k + VEC_SIZE), _mm256_load_pd(w + k + VEC_SIZE), sum2);
    }
    sum1 = _mm256_add_pd(sum1, sum2);
    REDUCE_ADD(sum1)
    _mm256_store_pd(tmp_arr, sum1);
    sum = tmp_arr[0];
    for (; k < i; k++) {
      sum += L[i * N + k] * w[k];
    }
    w[i] = (v[i] - sum) / L[i * N + i];
  }
  /* solve (L')^-1 with Gaussian elimination */
  for (i = n-1; i>= 0; i--) {
    w[i] /= L[i * N + i];
    __m256d w_i = _mm256_set1_pd(w[i]);
    for (k = 0; k + 2 * VEC_SIZE <= i; k += 2 * VEC_SIZE) {
      __m256d L_ik_vec = _mm256_load_pd(L + i * N + k);
      __m256d w_k = _mm256_load_pd(w + k);
      w_k = _mm256_sub_pd(w_k, _mm256_mul_pd(L_ik_vec, w_i));
      _mm256_store_pd(w + k, w_k);
      __m256d L_ik_4_vec = _mm256_load_pd(L + i * N + k + VEC_SIZE);
      __m256d w_k_4 = _mm256_load_pd(w + k + VEC_SIZE);
      w_k_4 = _mm256_sub_pd(w_k_4, _mm256_mul_pd(L_ik_4_vec, w_i));
      _mm256_store_pd(w + k + VEC_SIZE, w_k_4);
    }
    for (; k < i; k++) {
      w[k] -= L[i * N + k] * w[i];
    }
  }
}
#endif
