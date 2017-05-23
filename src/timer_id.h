#ifndef TIMER_ID_H
#define TIMER_ID_H

enum TIMER_ID{
  UPDATE_CHOLESKY,
  BACKSOLVE_CHOLESKY,

  END_ITR
};

static const char *TIMER_ID_STR[] = {
  "UPDATE_CHOLESKY",
  "BACKSOLVE_CHOLESKY",

  "END_ITR"
};

#endif
