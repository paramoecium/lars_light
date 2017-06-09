#include "rdtsc.h"
#include "assert.h"
#include "util.h"

#include "timer_id.h"

#ifndef TIMER_H
#define TIMER_H
class Timer{
public:
  Timer(int max_size_in);
  void reset();
  void start(TIMER_ID id);
  void end(TIMER_ID id);
  double get(TIMER_ID id);
  void print();
  void print(int num_runs);

private:
  double* clocks;
  tsc_counter* laps;
  int *call_cnt;
  int max_size;
};

inline Timer::Timer(int max_size_in) : max_size(max_size_in) {
  clocks = (double *) _mm_malloc(max_size * sizeof(double), 32);
  laps = (tsc_counter *) _mm_malloc(max_size * sizeof(tsc_counter), 32);
  call_cnt = (int *) _mm_malloc(max_size * sizeof(int), 32);
  
}

inline void Timer::reset() {
  memset(clocks, 0, max_size * sizeof(double));
  memset(call_cnt, 0, max_size * sizeof(int));
}

inline void Timer::start(TIMER_ID id) {
  assert(id < max_size);
  call_cnt[id] += 1;
  CPUID();
  RDTSC(laps[id]);
}

inline void Timer::end(TIMER_ID id) {
  assert(id < max_size);
  tsc_counter end;
  CPUID();
  RDTSC(end);

  clocks[id] += (double) (COUNTER_DIFF(end, laps[id]));
}

inline double Timer::get(TIMER_ID id) {
  assert(id < max_size);
  return clocks[id];
}

inline void Timer::print() {
  for (int i = 0; i < (int)END_ITR; i++) {
    printf("%s, %.3f\n",
      TIMER_ID_STR[i],
      call_cnt == 0? 0 : clocks[i] / call_cnt[i]);
  }
}

inline void Timer::print(int num_runs) {
  for (int i = 0; i < (int)END_ITR; i++) {
    printf("%s, %.3f\n",
      TIMER_ID_STR[i],
      call_cnt == 0? 0 : clocks[i] / num_runs);
  }
}

#endif
