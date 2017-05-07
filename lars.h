#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "util.h"
#include "mathOperations.h"
#include "cholesky.h"

#ifndef LARS_H
#define LARS_H

struct Lars {

  int N, p;
  int active_size, active_itr;

  const Real *Xt; //a pxN matrix, transpose of X(Nxp);
  const Real *y; //a 1xp vector
  Idx *beta, *beta_old; // current beta and old beta solution [Not sorted]

  int *active; // active[i] = position beta of active param or -1
  Real *c; //
  Real *w; // sign c[active_set]
  Real *u; // unit direction of each iteration
  Real *a; // store Xt * u
  Real *L; // lower triangular matrix of the gram matrix of X_A (pxp)

  Real lambda, lambda_new, lambda_old;

  Real *tmp; // temporary storage for active correlations

  FILE *fid;


  Lars(const Real *Xt_in, const Real *y_in, int p_in, int N_in, Real lambda_in);

  ~Lars();

  void solve();

  bool iterate();

//  void calculateParameters();

  void getParameters(Idx** beta_out) const; // get final beta

  Real compute_lambda(); // compute lambda given beta
};

Lars::Lars(const Real *Xt_in, const Real *y_in, int p_in, int N_in, Real lambda_in):
    Xt(Xt_in), y(y_in), p(p_in), N(N_in), lambda(lambda_in) {

  beta = (Idx*) calloc(p, sizeof(Idx));
  beta_old = (Idx*) calloc(p, sizeof(Idx));

  // Initializing
  active_size = fmin(N, p);
  active_itr = 0;
  active = (int*) malloc(active_size * sizeof(int));
  memset(active, -1, active_size * sizeof(int));
  c = (Real*) calloc(active_size, sizeof(Real));
  w = (Real*) calloc(active_size, sizeof(Real));
  u = (Real*) calloc(N, sizeof(Real));
  a = (Real*) calloc(active_size, sizeof(Real));
  L = (Real*) calloc(active_size * active_size, sizeof(Real));
  tmp = (Real*) calloc(p, sizeof(Real));

  mvm(Xt, false, y, c, p, N);

  fid = stderr;
}

Lars::~Lars() {
  print("Lars() DONE\n");
}

bool Lars::iterate() {
  if (active_itr > active_size) return false;

  Real C = 0.0;
  int cur = -1;
  for (int i = 0; i < p; ++i) {
    print("c[%d]=%.3f active[i]=%d\n", i, c[i], active[i]);
    if (active[i] != -1) continue;
    if (fabs(c[i]) > C) {
      cur = i;
      C = fabs(c[i]);
    }
  }
  // All remainging C are 0
  if (cur == -1) return false;

  print("Active set size is now %d\n", active_itr + 1);
  print("Activate %d column with %.3f value\n", cur, C);

  print("active[cur]=%d cur=%d p=%d\n", active[cur], cur, p);
  assert(active[cur] == -1 and active_itr < p);

  active[cur] = active_itr;
  beta[active_itr] = Idx(cur, 0);

  // calculate Xt_A * Xcur, Matrix * vector
  // new active row to add to gram matrix of active set
  for (int i = 0; i <= active_itr; ++i) {
      tmp[i] = dot(Xt + cur * N, Xt + beta[i].id * N, N);
  }
  // L[active_itr][] = tmp[];
  for (int i = 0; i <= active_itr; ++i) {
    L[active_itr*active_size + i] = tmp[i];
  }
  update_cholesky(L, active_itr, active_itr+1, active_size);
  print("L after cholesky\n");
  for (int i = 0; i <= active_itr; ++i) {
    for (int j = 0; j <= active_itr; ++j) {
      print("%.3f  ", L[i * active_size + j]);
    }
    print("\n");
  }
  print("\n");

  // set w[] = sign(c[])
  for (int i = 0; i <= active_itr; ++i) {
    w[i] = sign(c[beta[i].id]);
  }

  // w = R\(R'\s)
  // w is now storing sum of all rows? in the inverse of G_A
  backsolve(L, w, w, active_itr+1, active_size);

  // AA is is used to finalize w[]
  // AA = 1 / sqrt(sum of all entries in the inverse of G_A);
  Real AA = 0.0;
  for (int i = 0; i <= active_itr; ++i) {
    AA += w[i] * sign(c[beta[i].id]);
  }
  AA = 1.0 / sqrt(AA);
  print("AA: %.3f\n", AA);

  // get the actual w[]
  for (int i = 0; i <= active_itr; ++i) {
    w[i] *= AA;
  }
  print("w solved :");
  for (int i = 0; i < p; ++i) print("%.3f ", w[i]);
  print("\n");

  // get a = X' X_a w
  // Now do X' (X_a w)
  // can store a X' X_a that update only some spaces when adding new col to
  // activation set
  // Will X'(X_a w) be better? // more
  // X' (X_a w) more flops less space?
  // (X' X_a) w less flops more space?
  memset(a, 0, active_size*sizeof(Real));
  memset(u, 0, active_size*sizeof(Real));
  // u = X_a * w
  for (int i = 0; i <= active_itr; ++i) {
    axpy(w[i], &Xt[beta[i].id * N], u, p);
  }
  // a = X' * tmp
  mvm(Xt, false, u, a, p, N);

  print("u : ");
  for (int i = 0; i < N; i++) print("%.3f ", u[i]);
  print("\n");

  print("a : ");
  for (int i = 0; i < p; i++) print("%.3f ", a[i]);
  print("\n");

  Real gamma = C / AA;
  int gamma_id = cur;
  if (active_itr < active_size) {
    print("C=%.3f AA=%.3f\n", C, AA);
    for (int i = 0; i < p; i++) {
      if (active[i] != -1) continue;
      Real t1 = (C - c[i]) / (AA - a[i]);
      Real t2 = (C + c[i]) / (AA + a[i]);
      print("%d : t1 = %.3f, t2 = %.3f\n", i, t1, t2);

      if (t1 > 0 and t1 < gamma) gamma = t1, gamma_id=i;
      if (t2 > 0 and t2 < gamma) gamma = t2, gamma_id=i;
    }
  }
  print("gamma = %.3f from %d col\n", gamma, gamma_id);

  // add gamma * w to beta
  for (int i = 0; i <= active_itr; ++i)
    beta[i].v += gamma * w[i];

  // update correlation with a
  for (int i = 0; i < p; ++i)
    c[i] -= gamma * a[i];

  print("beta: ");
  for (int i = 0; i <= active_itr; ++i) print("%d %.3f ", beta[i].id, beta[i].v);
  print("\n");

  active_itr++;
  return true;
}

void Lars::solve() {
  int itr = 0;
  while (iterate()) {
    // compute lambda_new
    print("=========== The %d Iteration ends ===========\n", itr);

    //calculateParameters();
    lambda_new = compute_lambda();
    print("---------- lambda_new : %.3f lambda_old: %.3f lambda: %.3f\n", lambda_new, lambda_old, lambda);
    for (int i = 0; i < active_itr; i++)
      print("%d : %.3f %.3f\n", beta[i].id, beta[i].v, beta_old[i].v);

    if (lambda_new <= lambda) {
      lambda_old = lambda_new;
      memcpy(beta_old, beta, active_itr * sizeof(Idx));
    } else {
      Real factor = (lambda - lambda_old) / (lambda_new - lambda_old); //TODO use L1 norm
      for (int j = 0; j < active_itr; j++) {
//        beta[j].v = beta_old[j].v * (1.f - factor) + factor * beta[j].v;
        beta[j].v = beta_old[j].v + factor * (beta[j].v - beta_old[j].v);
      }
      break;
    }
    itr++;
  }
  print("LARS DONE\n");
}


//void Lars::calculateParameters() {
//  for (int i = 0; i < active_itr; i++) {
//    print("beta[%d] = %.3f\n", beta[i].id, beta[i].v);
//  }
//
//  mvm(Xt, false, y, tmp, p, N);
//
//  Real *tmp2 = (Real*) calloc(active_itr, sizeof(Real));
//  for (int i = 0; i < active_itr; ++i) {
//    tmp2[i] = tmp[beta[i].id];
//  }
//  backsolve(L, tmp2, tmp2, active_itr, active_size);
//
//  for (int i = 0; i < active_itr; ++i) {
//    beta[i].id = beta[i].id;
//    beta[i].v = tmp2[i];
//  }
//
//  free(tmp2);
//}

void Lars::getParameters(Idx** beta_out) const {
  *beta_out = beta;
}


// computes lambda given beta, lambda = max(abs(2*X'*(X*beta - y)))
inline Real Lars::compute_lambda() {
  memcpy(tmp, y, N * sizeof(Real));
  for (int i = 0; i < active_itr; i++) {
    for (int j = 0; j < N; j++)
      tmp[j] -= Xt[beta[i].id * N + j] * beta[i].v;
  }
  // compute X'*(y - X*beta)
  Real max_lambda = Real(0.0);
  mvm(Xt, false, tmp, tmp, p, N);
  for (int i = 0; i < p; ++i) {
    max_lambda = fmax(max_lambda, fabs(dot(Xt + i * N, tmp, N)));
  }
  return max_lambda;
}

#endif
