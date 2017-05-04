//-*-c++-*-
#ifndef DENSE_LARS_DATA_H
#define DENSE_LARS_DATA_H

/** class DenseLarsData<float/double>
 *
 * Main data class for Lars library.  Contains the matrix (X) and vector (y),
 * where we are finding regularized solutions to the equation X*beta = y.
 *
 * This class uses only flat C-style arrays internally, as this is
 * what is used by BLAS, which we use to do all the heavy lifting.
 *
 * This class is hidden behind the wrappers, which provide convenient
 * user interfaces to the library routines, so it is not likely that
 * any user will need to use this class.
 *
 * LARS++, Copyright (C) 2007 Varun Ganapathi, David Vickery, James
 * Diebel, Stanford University
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301 USA.
*/

// this header contains all cblas.h func
#include "replacementToC.h"
typedef float Real;

typedef enum {AUTO=-1, NO_KERN, KERN} KERNEL;

typedef struct Idx Idx;
struct Idx{
  int id;
  Real v;
}

typedef struct DenseLarsData DenseLarsData;
struct DenseLarsData {
private:
  // Main problem data
  const int N; // number of rows/samples
  const int p; // number of cols/features
  const Real* X;  // data matrix is Nxp
  const Real* y;  // response vector is Nx1

  // Internal work data
  Real* Xw;    // X*w is Nx1
  Real* Xty;   // X'*y is px1
  Real* tmp_p; // temporary storage px1

  // Kernelized LARS support
  KERNEL kernel;    // use the kernelized form by pre-computing X'*X
  bool precomputed; // the given data matrix is actually X'*X, not X
  Real** XtX_col;   // pointers to the columns of X'*X
  Real* XtX_buf;    // temp space only used when not precomputed

public:
  // Constructor that accepts flat C column-major arrays
  DenseLarsData(const Real* X_in, const Real* y_in, const int N_in, const int p_in,
    const KERNEL kernel_, const bool precomputed_);

  // Destructor
  ~DenseLarsData();

  // dots two column vectors together
  Real col_dot_product(int c1, int c2) const;

  // So that we can handle both kernelized and not
  void getXtY(Real* xty) const;
  void computeXtX();

  // Computes director correlation
  void compute_direction_correlation( 
          const Idx* beta, const int beta_size,
          const Real* wval, const int w_rows,
          Real* a);

  // Computes current lambda given a sparse vector beta
  Real compute_lamda(const Idx* beta, const int beta_size) const;

  // Computes least squares solution based on given sparsity pattern
  // LILY :: Not implemented?
  Real compute_lls_beta(Idx *beta) const;

  int nrows() const {return N;}
  int ncols() const {return p;}
}

DenseLarsData::DenseLarsData(const Real* X_in, const Real* y_in, const int N_in, const int p_in,
    const KERNEL kernel_, const bool precomputed_) :
    X(X_in), y(y_in), N(N_in), p(p_in), kernel(kernel_), precomputed(precomputed_) {
  
  // auto-select whether to use kernel or not
  if (kernel == AUTO) {
    if (N >= int(0.5*p)) kernel = KERN;
    else kernel = NO_KERN;
  }

  // allocate memory for internal work space
  tmp_p = (Real*) malloc(p * sizeof(Real);

  // - if X'*X and X'*y are precomputed, this implies we are using a kernel
  // - otherwise, we'll need some temp space to store X'*y
  if (precomputed == true) kernel = KERN;
  else Xty = (Real*) malloc(p * sizeof(Real));
  
  // - If we are using a kernelized form, we need XtX_col and possibly
  //   XtX_buf (if X'*X was not precomputed)
  // - otherwise we need space for X*w in computing X'*(X*w)
  if (kernel == KERN) {
    XtX_col = (Real**) malloc(p * sizeof(Real*));
    if (precomputed) {
      for (int i = 0; i < p; i++) {
        XtX_col[i] = (Real*) &X[i*p];
        XtX_buf = (Real*) X;
      }
    } else {
      XtX_buf = (Real*) malloc(p * p * sizeof(Real));
      computeXtX();
      for (int i = 0; i < p; i++) {
        XtX_col[i] = &XtX_buf[i*p];
      }
    }
  } else {
    Xw = (Real *) malloc(N * sizeof(Real));
  }
}

//Destructor
inline DenseLarsData::~DenseLarsData() {
  free(tmp_p);
  if (!precomputed) free(Xty);
  if (kernel == KERN) {
    free(XtX_col);
    if (!precomputed) free(XtX_buf);
  } else {
    free(Xw);
  }
}

///////////////////////////
// Double Precision Case //
///////////////////////////

// dots two column vectors together
inline Real DenseLarseData::col_dot_product(const int c1, const int c2) {
  if (kernel == KERN) {
    return XtX_buf[c1 + p*c2];
  } else {
    //return dot(N, &X[N*c1], 1, &X[N*c2], 1);
    // Dinesh: corrected
    dot(&X[N*c1], &X[N*c2], N);
  }
}

// So that we can handle both kernelized and not
inline void DenseLarseData::getXtY(Real* xty) const {
  free(xty);
  xty = (Real*) malloc(p * sizeof(Real);
  if (precomputed) {
    memcpy(xty, y, p * sizeof(Real));   
  } else {

    // LILY: I think we have to transpose X here?
    //matVecProd(X, y, Xty, N, p);
    // cblas_dgemv(CblasColMajor, CblasTrans, N, p, 1.0, X, N, y, 1, 0.0, Xty, 1);
    // Dinesh: this should be better
    mvm(X, true, y, Xty, N, p);
}

// Computes internal copy of X'*X if required
inline void DenseLarsData::computeXtX() {
  // LILY: I think we have to transpose X here?
  //gramMatrix(X, XtX, N, p);
  // Dinesh: corrected
  mmm(X, transpose, X, XtX_buf, p, N, p);
}

// Computes director correlation a = X'*X*w
inline void DenseLarsData::compute_direction_correlation(const Idx *beta, const int beta_size, 
    const Real* wval, const int w_rows, Real* a) {
  // clear old data
  memset(a, 0, p*sizeof(Real));

  // compute X'*X*beta, in one of two ways:
  if (kernel == KERN) {
    for (int i = 0, n = beta_size; i < n; ++i) {
      // add (X'*X)_i * w_i
      // cblas_daxpy(p, wval[i], XtX_col[beta[i].first], 1, a, 1);
      daxpy(wval[i], XtX_col[beta[i].Idx], a, p);
      // Dinesh: I am assuming here .Idx is your new variable name
    }
  } else {
    // add columns into X*w
    memset(Xw, 0, N*sizeof(Real));
    for (int i = 0, n = beta_size; i < n; ++i) {
      // cblas_daxpy(N, wval[i], &X[beta[i].first*N], 1, Xw, 1);
      daxpy(wval[i], &X[beta[i].Idx * N], Xw, N);
    }
    // now do X'*(X*w)
    // cblas_dgemv(CblasColMajor,CblasTrans,N,p,1.0,X,N,Xw,1,0.0,a,1);
    // LILY : X transpose?
    //matVecProd(X, Xw, a, N, p);
    // Dinesh: corrected
    mvm(X, true, Xw, a, N, p); 
  }
}

inline Real DenseLarsData::compute_lambda(const Idx* beta, const int beta_size) const {
  //clear old data
  memset(tmp_p, 0, p * sizeof(Real));

  // compute max(abs(2*X'*(X*beta - y))) in one of two ways:
  if (kernel == KERN) {
    // X'*y - (X'*X)*beta
    memcpy(tmp_p, Xty, p * sizeof(Real));
  
    for (int i = 0, n = beta_size; i < n; ++i) {
      // subtract (X'*X)_i * beta_i
      // cblas_daxpy(p, -beta[i].second, XtX_col[beta[i].first], 1, tmp_p, 1);
      daxpy(-beta[i].v, XtX_col[beta[i].id], tmp_p, p);
    }

    return 2.0 * fabs(tmp_p[ idamax(tmp_p, p)]);
  } else {
    // compute Xw = y - X*beta
    memcpy(Xw, y, N * sizeof(Real));
    for (int i = 0, n = beta_size; i < n; ++i) {
      daxpy(-beta[i].v, &X[N * beta[i].id], Xw, N);
    }
    
    // now compute 2*X'*Xw = 2*X'*(y - X*beta)
    // cblas_dgemv(CblasColMajor,CblasTrans,N,p,2.0,X,N,Xw,1,0.0,tmp_p,1);
    //scalarMutlplyVector(2, X, X2, tmp_p, N);
    amvm(2.0, X, true, Xw, tmp_p, N, p);
    return fabs(tmp_p[idamax(tmp_p, p)]);
  }
}

#endif
