#include <cstdio>

#include "src/util.h"
#include "src/lars.h"
#include "timer.h"

void set_value(const int D, const int K, Real *Xt, Real *y,
Real *beta) {
  prepare_Xt(D, K, true, Xt);
  prepare_Beta(K, 1, beta);
  memset(y, 0, sizeof(Real) * D);

  for (int i = 0; i < K; i++) {
    axpy(beta[i], &Xt[i * D], y, D);
  }

}

int main() {
  tsc_counter start, end;
  double cycles = 0.;
  size_t num_runs = 1;
  // Initailize data
  const int D = 1 << 7, K = 2 * D;
  //const int Max_D = 600, Max_K = 600;
  Real lambda = 0.0;
  Timer timer(END_ITR);
    
  Real *Xt = (Real*) malloc(sizeof(Real) * D * K);
  Real *y = (Real*) malloc(sizeof(Real) * D);
  Real *beta = (Real*) malloc(sizeof(Real) * K);
  Real *beta_h = (Real*) malloc(sizeof(Real) * K);
  set_value(D, K, Xt, y, beta);
  Lars lars(Xt, D, K, lambda, &timer);
    
  timer.reset();
  CPUID(); RDTSC(start);
  for (int i = 0; i < num_runs; ++i) {
      lars.set_y(y);
      lars.solve();
  }
  CPUID(); RDTSC(end);
  timer.print(num_runs);

  cycles = (double) (COUNTER_DIFF(end, start)) / num_runs;
  printf("TOTAL, %.3f\n\n", cycles);
  free(Xt);
  free(y);
  free(beta);
  free(beta_h);
}
