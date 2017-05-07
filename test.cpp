#include <cstdio>

#include "util.h"
#include "lars.h"

int main() {

  // Initailize data
  int D, K;
  Real *Xt;
  Real *y;
  Idx *beta;
  Real lambda = 4.5;

  D = 5, K = 2;
  Xt = (Real*) malloc(D * K * sizeof(Real));
  y = (Real*) malloc(D * sizeof(Real));
  beta = (Idx*) malloc(K * sizeof(Idx));

  Xt[0 * D + 0] = 1;
  Xt[0 * D + 1] = 2;
  Xt[0 * D + 2] = 1;
  Xt[0 * D + 3] = 1;
  Xt[0 * D + 4] = 1;
  
  Xt[1 * D + 0] = 1;
  Xt[1 * D + 1] = 1;
  Xt[1 * D + 2] = 0;
  Xt[0 * D + 3] = 1;
  Xt[0 * D + 4] = 1;

//  Xt[2 * D + 0] = 1;
//  Xt[2 * D + 1] = 1;
//  Xt[2 * D + 2] = 1;
//
//  Xt[3 * D + 0] = 0;
//  Xt[3 * D + 1] = 1;
//  Xt[3 * D + 2] = 0;
//
//  Xt[4 * D + 0] = 0;
//  Xt[4 * D + 1] = 0;
//  Xt[4 * D + 2] = 1;

  y[0] = 3;
  y[1] = 5;
  y[2] = 3;
  y[3] = 2;
  y[4] = 2;


  Lars lars(Xt, y, D, K, lambda);

  lars.solve();

  lars.getParameters(&beta);
  printf("get Parameters\n");

  for (int i = 0; i < K; i++)
    printf("%d : %.3f\n", beta[i].id, beta[i].v);
}
