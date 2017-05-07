#include <cstdarg>

#ifndef UTIL_H
#define UTIL_H

typedef double Real;

struct Idx {
  int id;
  Real v;

  Idx(int id_in, Real v_in): id(id_in), v(v_in) {}
};

inline Real sign(Real tmp) {
  if (tmp > 0) return 1.0;
  if (tmp < 0) return -1.0;
  return 0;
}


const bool DEBUG = false;
inline void print(const char *format, ...) {
  va_list arg;
  
  if (DEBUG) printf(format, arg);
}

#endif
