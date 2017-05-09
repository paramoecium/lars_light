#include "rdtsc.h"
#include "assert.h"

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

private:
  double* clocks;
  tsc_counter* laps;
  int *call_cnt;
  int max_size;
};

inline Timer::Timer(int max_size_in) : max_size(max_size_in) {
  clocks = (double *) calloc(max_size, sizeof(double));
  laps = (tsc_counter *) malloc(max_size * sizeof(tsc_counter));
  call_cnt = (int *) calloc(max_size, sizeof(int));
  
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
    printf("[TIMER] %s : %.3f cycles\n",
      TIMER_ID_STR[i],
      call_cnt == 0? 0 : clocks[i] / call_cnt[i]);
  }
}

#endif
