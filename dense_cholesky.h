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

struct DenseCholesky {

  Real *L; // lower triangular matrix
  int size = 0;
  const int max_size;

  /// Constructor accepts the maximum possible size of the cholesky
  DenseCholesky(int n) : max_size(n) {
    L = (Real *) malloc(max_size * max_size);
  }

  ~DenseCholesky() {
    free(L);
  }

  void addRowCol(const Real* vals);
  //void removeRowCol(int r);
  void solve(const Real *y, Real *beta);
  void print();
}

/// Add a new row/col to the internal matrix (the number of values expected
/// is equal to the number of existing rows/cols + 1)
void DenseCholesky::addRowCol(const Real* vals) {
  // grow L
  int j = size;
  size++;
  for (int i = 0; i <= j; ++i) {
    L[j * size + i] = vals[i];
  }
  update_cholesky(L, size, j);
}

/// Remove a row/column from X updates cholesky automatically
// void DenseCholesky::removeRowCol(int r) {
//   downdate_cholesky(L, size, r);
//   size--;
// }

/// Solves for beta given y
void DenseCholesky::solve(const Real *y, Real *beta) {
  backsolve(L, beta, y, size);
}

void DenseCholesky::print() {
  printf("[DenseCholesky] Used : %d\n", size);
  int i, j;
  for (i = 0; i < size; ++i) {
    for (j = 0; j < size; ++j) {
      printf("              %16.7le", L[i * size + j]);
    }
    printf("\n");
  }
  printf("\n");
};

#endif
