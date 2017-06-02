#include "rdtsc.h"
#include "assert.h"

#include "timer_id.h"

#ifndef TIMER_H
#define TIMER_H
class Timer{
public:
  Timer(int n_timer_in, int max_iter_in, int step_size_in);
  void reset();
  void start(TIMER_ID id, int iter);
  void end(TIMER_ID id, int iter);
  double get(TIMER_ID id);
  void print();
  void print(int num_runs);

private:
  double* clocks;
  tsc_counter* laps;
  int *call_cnt;
  int n_timer;

  int max_iter;
  int step_size;
  int n_record;
};

inline Timer::Timer(int n_timer_in, int max_iter_in, int step_size_in) : n_timer(n_timer_in), max_iter(max_iter_in), step_size(step_size_in) {
  n_record = max_iter/step_size;
  clocks = (double *) calloc(n_record * n_timer, sizeof(double));
  laps = (tsc_counter *) malloc(n_timer * sizeof(tsc_counter));
  call_cnt = (int *) calloc(n_record * n_timer, sizeof(int));
}

inline void Timer::reset() {
  memset(clocks, 0, n_record * n_timer * sizeof(double));
  memset(call_cnt, 0, n_record * n_timer * sizeof(int));
}

inline void Timer::start(TIMER_ID id, int iter) {
  if ((iter % step_size) > 0) return;
  assert(id < n_timer);
  call_cnt[(iter / step_size - 1) * n_timer + id] += 1;
  CPUID();
  RDTSC(laps[id]);
}

inline void Timer::end(TIMER_ID id, int iter) {
  if ((iter % step_size) > 0) return;
  assert(id < n_timer);
  tsc_counter end;
  CPUID();
  RDTSC(end);
  clocks[(iter / step_size - 1) * n_timer + id] += (double) (COUNTER_DIFF(end, laps[id]));
}

inline void Timer::print() {
  for (int t = 0; t < n_record; t++) {
    printf("iteration %d:\n", (t + 1) * step_size);
    for (int i = 0; i < (int)END_ITR; i++) {
      printf("%s, %.3f\n",
        TIMER_ID_STR[i],
        call_cnt == 0? 0 : clocks[t * n_timer + i] / call_cnt[t * n_timer + i]);
    }
    printf("\n");
  }
}

#endif
