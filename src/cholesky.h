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
const int B = 512; //block size;

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
  Real sum;
  __m256d tmp0, tmp1, tmp2, tmp3; //for macros
  Real tmp_arr[VEC_SIZE];
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
      tmp0 = _mm256_setzero_pd();
      for (k = b; k <= i - VEC_SIZE; k += VEC_SIZE) {
        tmp0 = _mm256_fmadd_pd(_mm256_load_pd(L + i * N + k), _mm256_load_pd(L + j * N + k), tmp0);
      }
      REDUCE_ADD(tmp0)
      _mm256_store_pd(tmp_arr, tmp0);
      sum = tmp_arr[0];
      for (; k < i; ++k) {
        sum += L[i * N + k] * L[j * N + k];
      }
      L[j * N + i] = (L[j * N + i] - sum) / L[i * N + i];
    }
    /* solving the rectangle below */
    for (;i < j; i++) {
      tmp0 = _mm256_setzero_pd();
      for (k = b; k <= b + B - VEC_SIZE; k += VEC_SIZE) { // B is a multiple of VEC_SIZE
        tmp0 = _mm256_fmadd_pd(_mm256_load_pd(L + i * N + k), _mm256_load_pd(L + j * N + k), tmp0);
      }
      REDUCE_ADD(tmp0)
      _mm256_store_pd(tmp_arr, tmp0);
      L[j * N + i] -= tmp_arr[0];
    }
  }
  /* finish the remaining triangle */
  for (i = b; i < j; i++) {
    tmp0 = _mm256_setzero_pd();
    for (k = b; k <= i - VEC_SIZE; k += VEC_SIZE) {
      tmp0 = _mm256_fmadd_pd(_mm256_load_pd(L + i * N + k), _mm256_load_pd(L + j * N + k), tmp0);
    }
    REDUCE_ADD(tmp0)
    _mm256_store_pd(tmp_arr, tmp0);
    sum = tmp_arr[0];
    for (; k < i; ++k) {
      sum += L[i * N + k] * L[j * N + k];
    }
    L[j * N + i] = (L[j * N + i] - sum) / L[i * N + i];
  }
  /* compute the lower right entry */
  tmp0 = _mm256_setzero_pd();
  for (k = 0; k <= j - VEC_SIZE; k += VEC_SIZE) {
    __m256d L_j_N_k = _mm256_load_pd(L + j * N + k);
    tmp0 = _mm256_fmadd_pd(L_j_N_k, L_j_N_k, tmp0);
  }
  REDUCE_ADD(tmp0)
  _mm256_store_pd(tmp_arr, tmp0);
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
