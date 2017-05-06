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

  const Real *Xt; //a Nxp matrix, transpose of X;
  const Real *y; //a 1xp vector
  Idx *beta; // current parameters(solution) [Not sorted] a 1xp vector

  int *active; // active[i] = position in beta of active param or -1
  Real *c; // 
  Real *w; // sign c[active_set]
  Real *a; // store Xt * u
  Real *L; // lower triangular matrix of the gram matrix of X_A (pxp)

  Real *tmp; // temporary storage for active correlations

  FILE *fid;


  void getParameters(Idx **beta_out) const; // get current beta

  Lars(const Real *Xt_in, const Real *y_in, int p_in, int N_in);

  ~Lars();

  bool iterate();
};

Lars::Lars(const Real *Xt_in, const Real *y_in, int p_in, int N_in): 
    Xt(Xt_in), y(y_in), p(p_in), N(N_in){
  
  beta = (Idx*) calloc(p, sizeof(Idx));

  // Initializing
  active_size = fmin(N-1, p);
  active_itr = 0;
  active = (int*) malloc(active_size * sizeof(int));
  memset(active, -1, active_size * sizeof(int));
  c = (Real*) calloc(active_size, sizeof(Real));
  w = (Real*) calloc(active_size, sizeof(Real));
  a = (Real*) calloc(active_size, sizeof(Real));
  L = (Real*) calloc(active_size, active_size * sizeof(Real));
  tmp = (Real*) calloc(N, sizeof(Real));

  mvm(Xt, false, y, c, p, N);

  fid = stderr;
}

Lars::~Lars() {
  fprintf(fid, "Lars() DONE\n");
}

bool Lars::iterate() {
  printf("_------------------\n");
  if (active_itr >= active_size) return false;
  
  Real C = 0.0;
  int cur = -1;
  for (int i = 0; i < p; ++i) {
    printf("c[%d]=%.3f\n", i, c[i]);
    if (active[i] != -1) continue;
    if (fabs(c[i]) > C) {
      cur = i;
      C = fabs(c[i]);
    }
  }
  // All remainging C are 0
  if (cur == -1) return false;

  fprintf(fid, "Active set size is now %d\n", active_itr + 1);
  fprintf(fid, "Activate %d column with %.3f value\n", cur, C);

  printf("%d %d %d %d\n", active[cur], cur, active_itr, p);
  assert(active[cur] == -1 and active_itr < p);

  active[cur] = active_itr;
  beta[active_itr] = Idx(cur, 0);

  // calculate Xt_A * Xcur, Matrix * vector
  // new active row to add to gram matrix of active set
  printf("active itr%d\n", active_itr);
  for (int i = 0; i <= active_itr; ++i) {
    tmp[i] = dot(Xt + cur * N, Xt + beta[i].id * N, N); 
    printf("dot %d %d cols\n", cur, beta[i].id);
  }
  // L[active_itr][] = tmp[];
  for (int i = 0; i <= active_itr; ++i) {
    L[active_itr*p + i] = tmp[i];
  }
  update_cholesky(L, active_itr+1, active_itr);
  printf("L\n");
  for (int i = 0; i <= active_itr; ++i) {
    for (int j = 0; j <= active_itr; ++j) {
      printf("%.3f  ", L[i * (active_itr + 1) + j]);
    }
    printf("\n");
  }
  printf("\n");

  // set w[] = sign(c[])
  for (int i = 0; i <= active_itr; ++i) {
    w[i] = sign(c[beta[i].id]);
  }
  // w = R\(R'\s)
  // w is now storing sum of all rows? in the inverse of G_A
  backsolve(L, w, w, active_itr+1);

  // AA is is used to finalize w[]
  // AA = 1 / sqrt(sum of all entries in the inverse of G_A);
  Real AA = 0.0;
  for (int i = 0; i <= active_itr; ++i) {
    printf("%d w[i]%.3f beta[i]%.3f\n", i, w[i], c[beta[i].id]); 
    AA += w[i] * sign(c[beta[i].id]);
  }
  AA = 1.0 / sqrt(AA);
  fprintf(fid, "AA: %.3f\n", AA);

  // get the actual w[]
  for (int i = 0; i <= active_itr; ++i) {
    w[i] *= AA;
  }

  // get a = X' X_a w
  // Now do X' (X_a w)
  // can store a X' X_a that update only some spaces when adding new col to
  // activation set
  // Will X'(X_a w) be better? // more 
  // X' (X_a w) more flops less space?
  // (X' X_a) w less flops more space?
  memset(tmp, 0, N*sizeof(Real));
  memset(a, 0, active_size*sizeof(Real));
  // tmp = X_a * w
  for (int i = 0; i <= active_itr; ++i) {
    daxpy(1.0, &Xt[beta[i].id * N], tmp, N);
  }
  // a = X' * tmp
  mvm(Xt, false,tmp, a, p, N); 

  printf("a : ");
  for (int i = 0; i < p; i++) printf("%.3f ", a[i]);
  printf("\n");

  Real gamma = C / AA;
  if (active_itr < active_size) {
    for (int i = 0; i <= active_itr; i++) {
      int j = beta[i].id;
      Real t1 = (C - c[j]) / (AA - a[j]);
      Real t2 = (C + c[j]) / (AA + a[j]);

      if (t1 > 0 and t1 < gamma) gamma = t1;
      if (t2 > 0 and t2 < gamma) gamma = t2;
    }
  }
  fprintf(fid, "gamma = %.3f\n", gamma);

  // add lambda * w to beta
  for (int i = 0; i <= active_itr; ++i)
    beta[i].v += gamma * w[i];

  // update correlation with a
  for (int i = 0; i <= active_itr; ++i)
    c[i] -= gamma * a[i];

  fprintf(fid, "beta: ");
  for (int i = 0; i <= active_itr; ++i) fprintf(fid, "%d %.3f ", beta[i].id, beta[i].v);
  fprintf(fid, "\n");

  active_itr++;
  return true;
}


void Lars::getParameters(Idx** beta_out) const {
  *beta_out = beta; 
}

#endif
