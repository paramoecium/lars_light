#ifndef TIMER_ID_H
#define TIMER_ID_H

enum TIMER_ID{
  GET_ACTIVE_IDX,
  FUSED_CHOLESKY,
  END_ITR
};

static const char *TIMER_ID_STR[] = {
  "GET_ACTIVE_IDX",
  "FUSED_CHOLESKY",

  "END_ITR"
};

#endif
