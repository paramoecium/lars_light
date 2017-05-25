#ifndef TIMER_ID_H
#define TIMER_ID_H

enum TIMER_ID{
  UPDATE_GRAM_MATRIX,
  FUSED_CHOLESKY,
  GET_AA,

  END_ITR
};

static const char *TIMER_ID_STR[] = {
  "UPDATE_GRAM_MATRIX",
  "FUSED_CHOLESKY",
  "GET_AA",

  "END_ITR"
};

#endif
