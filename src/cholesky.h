//-*-c++-*-
#ifndef CHOLESKY_H
#define CHOLESKY_H

#include <numeric>
#include <cstdio>
#include <cmath>
#include <immintrin.h>

#include "util.h"

const Real EPSILON = 1e-9;
#define REDUCE_ADD(target){\
tmp1 = _mm256_permute_ps(target, 0b0101);\
tmp2 = _mm256_add_ps(target, tmp1);\
tmp3 = _mm256_permute2f128_ps(tmp2, tmp2, 0b00000001);\
target = _mm256_add_ps(tmp2, tmp3);\
}
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
  float sum;
  float tmp_arr[8];
  float eps_small = EPSILON;
  int i, k;
  __m256 tmp0, tmp1, tmp2, tmp3; //for macros
  /* solve L^-1 with Gaussian elimination */
  for (i = 0; i < j; ++i) {
    tmp0 = _mm256_setzero_ps();
    for (k = 0; k + 8 <= i; k+=8) {
      __m256 L_ik_8 = _mm256_load_ps(L + i * N + k);
      __m256 L_jk_8 = _mm256_load_ps(L + j * N + k);
      tmp0 = _mm256_fmadd_ps(L_ik_8, L_jk_8, tmp0);
    }
    REDUCE_ADD(tmp0)
    _mm256_store_ps(tmp_arr, tmp0);
    sum = tmp_arr[0];
    for (; k < i; ++k) {
      sum += L[i * N + k] * L[j * N + k];
    }
    L[j * N + i] = (L[j * N + i] - sum) / L[i * N + i];
  }
  /* computer the lower right entry */
  sum = L[j * N + j];
  tmp0 = _mm256_setzero_ps();
  for (k = 0; k + 8 <= j; k+=8) {
    __m256 L_jk_8 = _mm256_load_ps(L + j * N + k);
    tmp0 = _mm256_fmadd_ps(L_jk_8, L_jk_8, tmp0);
  }
  REDUCE_ADD(tmp0)
  _mm256_store_ps(tmp_arr, tmp0);
  sum -= tmp_arr[0];
  for (; k < j; k++) {
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
  float sum;
  float tmp_arr[8];
  int i, k;
  __m256 tmp0, tmp1, tmp2, tmp3; //for macros
  /* solve L^-1 with Gaussian elimination */
  for (i = 0; i < n; i++) {
    tmp0 = _mm256_setzero_ps();
    for (k = 0; k + 8 <= i; k+=8) {
      __m256 w_k_8 = _mm256_load_ps(w + k);
      __m256 L_ik_8 = _mm256_load_ps(L + i * N + k);
      tmp0 = _mm256_fmadd_ps(w_k_8, L_ik_8, tmp0);
    }
    REDUCE_ADD(tmp0)
    _mm256_store_ps(tmp_arr, tmp0);
    sum = tmp_arr[0];
    for (; k < i; ++k) {
      sum += L[i * N + k] * w[k];
    }
    w[i] = (v[i] - sum) / L[i * N + i];
  }

  /* solve (L')^-1 with Gaussian elimination */
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
