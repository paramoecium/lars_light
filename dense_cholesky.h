//-*-c++-*-
#ifndef __DENSE_CHOLESKY_H
#define __DENSE_CHOLESKY_H

/** class DenseCholesky<float/double>
 *
 * Represents state of cholesky factorization of symmetric positive definite X
 * - Allows you to add row/col to X and incrementally update cholesky
 * - Allows you to remove row/col from X and incrementally update cholesky
 * - Allows you to use the cholesky factorization to
 * - find beta for X beta = y
 *
 * Uses general utility routines for cholesky decompositions found in
 * cholesky.h
 *
 * LARS++, Copyright (C) 2007 Varun Ganapathi, David Vickery, James
 * Diebel, Stanford University
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301 USA.
*/

//#include <iostream>
#include <cstdio>
#include "cholesky.h"

//using namespace std;
typedef float Real;

typedef struct DenseCholesky DenseCholesky;
struct DenseCholesky {

  Real *A;
  int used;
  int A_rows, A_cols;
  int max_size;

  /// Constructor accepts the maximum possible size of the cholesky
  DenseCholesky(int max_size) {
    max_size = max_size;
    used = A_rows = A_cols = 0;
    A = (Real *) malloc(max_size * max_size);
  }

  void (*_resize) (DenseCholesky *, int *);
  void (*addRowCol) (DenseCholesky *, const Real*);
  void (*removeRowCol) (DenseCholesky *, int);
  void (*solve) (DenseCholesky *, const Real*, Real*);
}

void _resize(DenseCholesky *self, int howManyRowCol) {
  if (self->A_rows < howManyRowCol || self->A_cols < howManyRowCol) {
    assert(0);
  }
  used = howManyRowCol;
}

/// Add a new row/col to the internal matrix (the number of values expected
/// is equal to the number of existing rows/cols + 1)
void addRowCol(DenseCholesky *self, const Real* vals) {
  // grow A
  int j = used, i;
  self->_resize(self, used + 1);
  for (i = 0; i <= j; ++i) {
    self->A[j * self->A_cols + i] = vals[i];
  }
  update_cholesky(self->A, self->used, j);
}

/// Remove a row/column from X updates cholesky automatically
void removeRowCol(DenseCholesky *self, int r) {
  downdate_cholesky(self->A, self->used, r);
  self->used -= 1;
  self->A_cols = self->A_rows = self->used;
}

/// Solves for beta given y
void solve(DenseCholesky *self, const Real *y, Real *beta) {
  // nvars is found in lars.h?
  y_copy = (Real*)calloc(nvars, sizeof(Real));
  memcpy(y_copy, y, sizeof(y));
  backsolve(self->A, beta, y, self->used);
  free(y_copy);
}

void print(DenseCholesky *self) {
  printf("[DenseCholesky] Used : %d\n", self->used);
  int i, j;
  for (i = 0; i < used; ++i) {
    for (j = 0; j < used; ++j) {
      printf("              %16.7le", self->A[i * self->A_cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}

#endif
