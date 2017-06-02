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
  timer.print();

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
  const int D = 1 << 10, K = 2 * D;
  Real lambda = 0.0;

  Real *Xt = (Real*) malloc(sizeof(Real) * D * K);
  Real *y = (Real*) malloc(sizeof(Real) * D);
  Real *beta = (Real*) malloc(sizeof(Real) * K);
  Real *beta_h = (Real*) malloc(sizeof(Real) * K);

  printf("\nD = %d, K = %d\n", D , K);
  Timer timer(END_ITR, D, D/8);
  timer.reset();
  set_value(D, K, Xt, y, beta);
  int num_runs = measure(D, K, Xt, y, beta, beta_h, lambda, timer);
}
