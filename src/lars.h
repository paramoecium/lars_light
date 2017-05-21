#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstring>

#include "util.h"
#include "mathOperations.h"
#include "cholesky.h"
#include "timer.h"

#ifndef LARS_H
#define LARS_H

struct Lars {

  int K, D;
  int active_size, active_itr;

  const Real *Xt; //a KxD matrix, transpose of X;
  const Real *y; //a Dx1 vector
  Idx *beta, *beta_old; // current beta and old beta solution [Not sorted]

  int *active; // active[i] = position beta of active param or -1
  Real *c; //
  Real *w; // sign c[active_set]
  Real *u; // unit direction of each iteration
  Real *a; // store Xt * u
  Real *L; // lower triangular matrix of the gram matrix of X_A (pxp)

  Real lambda, lambda_new, lambda_old;

  Real *tmp; // temporary storage for active correlations

  Timer timer;

  // allocate all needed memory
  // input fixed numbers
  Lars(const Real *Xt_in, int D_in, int K_in, Real lambda_in, Timer &timer_in);

  ~Lars();

  // input y for computation
  void set_y(const Real *y_in);

  void solve();

  bool iterate();

//  void calculateParameters();

  void getParameters(Idx** beta_out) const; // get final beta

  void getParameters(Real* beta_out) const;

  Real compute_lambda(); // compute lambda given beta
};

#endif
