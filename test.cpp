#include <cstdio>

#include "src/util.h"
#include "src/lars.h"
#include "timer.h"

int main() {

  // Initailize data
  int D, K;
  Real *Xt;
  Real *y;
  Idx *beta;
  Real lambda = 0.0;
  Timer timer(END_ITR);

  D = 3, K = 3;
  Xt = (Real*) malloc(D * K * sizeof(Real));
  y = (Real*) malloc(D * sizeof(Real));

//  prepareData(D, K, 1, true, Xt, y);

  Xt[0 * D + 0] = 0.043;
  Xt[0 * D + 1] = -0.728;
  Xt[0 * D + 2] = 0.685;

  Xt[1 * D + 0] = -.265;
  Xt[1 * D + 1] = 0.801;
  Xt[1 * D + 2] = -0.536;

  Xt[2 * D + 0] = -0.572;
  Xt[2 * D + 1] = -.218;
  Xt[2 * D + 2] = 0.790;

  y[0] = -0.199372;
  y[1] = -0.723805;
  y[2] = 0.923177;

  Lars lars(Xt, D, K, lambda, timer);

  lars.set_y(y);

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
  Real *b = (Real*)malloc(D * sizeof(Real));
  lars.getParameters(b);
  printf("error2 = %.3f\n", sqrt(get_square_error(Xt, b, y, D)));
}
