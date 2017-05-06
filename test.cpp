#include <cstdio>

#include "util.h"
#include "lars.h"

int main() {

  // Initailize data
  int N, p;
  Real *Xt;
  Real *y;
  Idx *beta;

  N = 3, p = 2;
  Xt = (Real*) malloc(N * p * sizeof(Real));
  y = (Real*) malloc(N * sizeof(Real));

  Xt[0 * N + 0] = -1;
  Xt[0 * N + 1] = 0;
  Xt[0 * N + 2] = 1;
  Xt[1 * N + 0] = 1;
  Xt[1 * N + 1] = 0;
  Xt[1 * N + 2] = 1;

  y[0] = -1.1111;
  y[1] = 0;
  y[2] = -1.1111;


  Lars lars(Xt, y, p, N);

  for (int itr = 0; lars.iterate(); itr++) {
    printf("==== The %d Iteration ends\n", itr);
  }
  printf("LARS DONE\n");

  lars.getParameters(&beta);
  printf("get parameters\n");

  for (int i = 0; i < p; i++)
    printf("%d : %.3f\n", beta[i].id, beta[i].v);
}
