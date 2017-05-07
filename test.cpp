#include <cstdio>

#include "util.h"
#include "lars.h"

int main() {

  // Initailize data
  int D, K;
  Real *Xt;
  Real *y;
  Idx *beta;
  Real lambda = 0.1;

  D = 6, K = 3;
  Xt = (Real*) malloc(D * K * sizeof(Real));
  y = (Real*) malloc(D * sizeof(Real));
  beta = (Idx*) malloc(K * sizeof(Idx));

  prepareData(D, K, 1, true, Xt, y);
  /*
  Xt[0 * D + 0] = 1;
  Xt[0 * D + 1] = 2;
  Xt[0 * D + 2] = 1;
  Xt[1 * D + 0] = 1;
  Xt[1 * D + 1] = 1;
  Xt[1 * D + 2] = 0;
  Xt[2 * D + 0] = 1;
  Xt[2 * D + 1] = 0;
  Xt[2 * D + 2] = 1;


  y[0] = 5;
  y[1] = 2;
  y[2] = 4;
  */

  Lars lars(Xt, y, D, K, lambda);

  lars.solve();

  lars.getParameters(&beta);
  printf("get Parameters\n");

  for (int i = 0; i < lars.active_itr; i++)
    printf("%d : %.3f\n", beta[i].id, beta[i].v);

  for (int i = 0; i < lars.active_itr; i++) {
    for (int j = 0; j < D; j++)
      y[j] -= Xt[beta[i].id * D + j] * beta[i].v;
  }
  Real sqr_error = Real(0.0);
  for (int j = 0; j < D; j++)
    sqr_error += y[j] * y[j];
  printf("error = %.3f\n", sqrt(sqr_error));
}
