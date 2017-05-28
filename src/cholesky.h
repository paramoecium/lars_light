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
//#define _mm256_load_pd _mm256_loadu_pd
//#define _mm256_store_pd _mm256_storeu_pd //TODO check memory alignment

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
const int B = 1024; //block size, multiple of 16;
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
inline Real update_cholesky_n_solve(Real *L, Real *w, const Real *v, const int n, const int N,
                  const Real *Xt, const int cur, const Idx *beta, const int D) {
  __m256d sum_v1, sum_v2, sum_v3, sum_v4, sum_v5, sum_v6, sum_v7, sum_v8;
  __m256d tmp0, tmp1, tmp2, tmp3; //for macros
  Real tmp_arr[2 * VEC_SIZE];
  Real sum_s1, sum_s2, sum_s3, sum_s4;

  int i, k, b_i, b_k;
  const Real *Xt_cur = Xt + cur * D;
  for (b_i = 0; b_i + B <= n; b_i += B) {
    /* initialize L(n, i) and w[i] */
    for (i = b_i; i < b_i + B; i++) { // TODO Lily
      L(n, i) = 0.0;
      for (int k = 0; k < D; k++) {
        L(n, i) += Xt[cur * D + k] * Xt[beta[i].id * D + k];
      }
      w[i] = v[i];
    }
    /* compute mvms (BxB)(Bx1), avx and unroll4 */
    for (b_k = 0; b_k + B <= b_i; b_k += B) {
      for (i = b_i; i < b_i + B; i++) {
        sum_v1 = _mm256_setzero_pd();
        sum_v2 = _mm256_setzero_pd();
        sum_v3 = _mm256_setzero_pd();
        sum_v4 = _mm256_setzero_pd();
        sum_v5 = _mm256_setzero_pd();
        sum_v6 = _mm256_setzero_pd();
        sum_v7 = _mm256_setzero_pd();
        sum_v8 = _mm256_setzero_pd();
        for (k = b_k; k <= b_k + B - 4 * VEC_SIZE; k += 4 * VEC_SIZE) {
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
        L(n, i) -= sum_s1;
        w[i] -= sum_s2;
      }
    }
    /* pivot the triangle, unroll2 */
    for (i = b_i; i < b_i + B; i++) {
      sum_s1 = 0.0, sum_s2 = 0.0, sum_s3 = 0.0, sum_s4 = 0.0;
      for (k = b_k; k <= i - 2; k += 2) {
        sum_s1 += L(i, k) * L(n, k);
        sum_s3 += L(i, k + 1) * L(n, k + 1);
        sum_s2 += L(i, k) * w[k];
        sum_s4 += L(i, k + 1) * w[k + 1];
      }
      for (; k < i; ++k) {
        sum_s1 += L(i, k) * L(n, k);
        sum_s2 += L(i, k) * w[k];
      }
      L(n, i) = (L(n, i) - sum_s1 - sum_s3) / L(i, i);
      w[i] = (w[i] - sum_s2 - sum_s4) / L(i, i);

    }
  }
  /* finish the remaining trapezoid */
  /* solve (L^-1)X'v and (L^-1)w with Gaussian elimination */
  for (i = b_i; i < n; i++) {
    sum_v1 = _mm256_setzero_pd();
    sum_v2 = _mm256_setzero_pd();
    sum_v3 = _mm256_setzero_pd();
    sum_v4 = _mm256_setzero_pd();
    sum_v5 = _mm256_setzero_pd();
    sum_v6 = _mm256_setzero_pd();
    sum_v7 = _mm256_setzero_pd();
    sum_v8 = _mm256_setzero_pd();
    L(n, i) = 0.0;
    for (int k = 0; k < D; k++) {
      L(n, i) += Xt[cur * D + k] * Xt[beta[i].id * D + k];
    }
    for (k = 0; k <= i - 4 * VEC_SIZE; k += 4 * VEC_SIZE) {
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
    for (; k < i; ++k) { // less than 16 times, unroll?
      sum_s1 += L(i, k) * L(n, k);
      sum_s2 += L(i, k) * w[k];
    }
    L(n, i) = (L(n, i) - sum_s1) / L(i, i);
    w[i] = (v[i] - sum_s2) / L(i, i);
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
  for (k = 0; k <= n - 4 * VEC_SIZE; k += 4 * VEC_SIZE) {
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
  for (; k < n; k++) { // less than 16 times, unroll?
    sum_s1 += L(n, k) * L(n, k);
    sum_s2 += L(n, k) * w[k];
  }


  Real L_n_n = 0.0;
  for (k = 0; k < D; k++) {
    L_n_n += Xt[cur * D + k] * Xt[beta[n].id * D + k];
  }
  L_n_n = sqrt(fmax(L_n_n - sum_s1, EPSILON));
  w[n] = (v[n] - sum_s2) / L_n_n;
  L(n, n) = L_n_n;

  Real AA = 0.0;
  /* solve ((L')^-1)w with Gaussian elimination */
  for (b_i = n; b_i - B >= -1; b_i -= B) {
    /* pivot the triangle, unroll2 */
    for (i = b_i; i > b_i - B; i--) {
      w[i] /= L(i, i);
      AA += w[i] * v[i];
      for (k = b_i - B + 1; k <= i - 2 ; k += 2) {
        w[k] -= L(i, k) * w[i];
        w[k + 1] -= L(i, k + 1) * w[i];
      }
      for (; k < i ; k++) {
        w[k] -= L(i, k) * w[i];
      }
    }
    /* compute mvms (BxB)(Bx1) */
    for (b_k = b_i - B; b_k - B >= -1; b_k -= B) {
      for (i = b_i; i > b_i - B; i--) {
        __m256d w_i = _mm256_set1_pd(w[i]);
        for (k = b_k - B + 1; k <= b_k - 4 * VEC_SIZE + 1; k += 4 * VEC_SIZE) {
          Real *L_i_k = &L(i, k);
          __m256d w_k = _mm256_load_pd(w + k);
          __m256d w_k_4 = _mm256_load_pd(w + k + 4);
          __m256d w_k_8 = _mm256_load_pd(w + k + 8);
          __m256d w_k_12 = _mm256_load_pd(w + k + 12);
          w_k = _mm256_fnmadd_pd(_mm256_load_pd(L_i_k), w_i, w_k);
          w_k_4 = _mm256_fnmadd_pd(_mm256_load_pd(L_i_k + 4), w_i, w_k_4);
          w_k_8 = _mm256_fnmadd_pd(_mm256_load_pd(L_i_k + 8), w_i, w_k_8);
          w_k_12 = _mm256_fnmadd_pd(_mm256_load_pd(L_i_k + 12), w_i, w_k_12);
          _mm256_store_pd(w + k, w_k);
          _mm256_store_pd(w + k + 4, w_k_4);
          _mm256_store_pd(w + k + 8, w_k_8);
          _mm256_store_pd(w + k + 12, w_k_12);
        }
      }
    }
    /* finish the remaining rectangle */
    for (i = b_i; i > b_i - B; i--) {
      __m256d w_i = _mm256_set1_pd(w[i]);
      for (k = 0; k <= b_k - 2 * VEC_SIZE + 1; k += 2 * VEC_SIZE) { // less than B
        Real *L_i_k = &L(i, k);
        __m256d w_k = _mm256_load_pd(w + k);
        __m256d w_k_4 = _mm256_load_pd(w + k + 4);
        w_k = _mm256_fnmadd_pd(_mm256_load_pd(L_i_k), w_i, w_k);
        w_k_4 = _mm256_fnmadd_pd(_mm256_load_pd(L_i_k + 4), w_i, w_k_4);
        _mm256_store_pd(w + k, w_k);
        _mm256_store_pd(w + k + 4, w_k_4);
      }
      for (; k <= b_k; k++) {
        w[k] -= L(i, k) * w[i];
      }
    }
  }
  /* finish the remaining triangle */
  for (; i>= 0; i--) {
    w[i] /= L(i, i);
    AA += w[i] * v[i];
    for (k = 0; k <= i - 2; k += 2) {
      w[k] -= L(i, k) * w[i];
      w[k + 1] -= L(i, k + 1) * w[i];
    }
    for (; k < i; k++) {
      w[k] -= L(i, k) * w[i];
    }
  }
  AA = 1.0 / sqrt(AA);
  return AA;
}
#endif
