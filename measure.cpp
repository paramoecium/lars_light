#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

#include <x86intrin.h>

#include "util.h"
#include "lars.h"
#include "mathOperations.h"
#include "rdtsc.h"
#include "timer.h"

#define RUNS 2
#define CYCLES_REQUIRED 1e7
#define VERIFY

int measure(const int D, const int K, Real *Xt, Real *y, Real *beta, Real *beta_h, Real lambda, Timer timer) {
  tsc_counter start, end;
  double cycles = 0.;
  size_t num_runs = RUNS;

  CPUID(); RDTSC(start); CPUID(); RDTSC(end);
  CPUID(); RDTSC(start); CPUID(); RDTSC(end);
  CPUID(); RDTSC(start); CPUID(); RDTSC(end);

  // Warm-up phase: determine number of runs needed
  // to ignore the timing overhead
  Lars lars(Xt, D, K, lambda, timer);

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

  #ifdef VERIFY
    lars.getParameters(beta_h);
    Real sqr_err = get_square_error(Xt, beta_h, y, D, K);
    if (sqr_err > 1e-5 or sqr_err != sqr_err)
      printf("\nVALIDATION FAILED: get error %.3f in lars with lambda %.3f\n\n", sqr_err, lambda);
  #endif

  return num_runs;
}

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
  const int Max_D = 1 << 12, Max_K = 2 * Max_D;
  //const int Max_D = 600, Max_K = 600;
  Real lambda = 0.0;
  Timer timer(END_ITR);

  Real *Xt = (Real*) malloc(sizeof(Real) * Max_D * Max_K);
  Real *y = (Real*) malloc(sizeof(Real) * Max_D);
  Real *beta = (Real*) malloc(sizeof(Real) * Max_K);
  Real *beta_h = (Real*) malloc(sizeof(Real) * Max_K);

  for (int i = 1 << 7; i <= Max_D; i <<=1) {
    printf("\nD = %d, K = %d\n", i , 2 * i);
    timer.reset();
    set_value(i, 2 * i, Xt, y, beta);
    int num_runs = measure(i, 2 * i, Xt, y, beta, beta_h, lambda, timer);

  }
}
