#ifndef LARS__H
#define LARS__H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <valarray>

#include "cholesky.h"

using namespace std;
typedef float Real;

struct Idx {
  int id;
  Real v;
}

struct Lars {
  DenseLarsData *data_; // data(contains X and y)
  beta_pair *beta_;   // current parameters(solution) [Not Sorted]

  // incrementally updated quantities
  int *active_; // active[i] = position in beta of active param or -1
  Real *c_; // correlation of columns of X with current residual
  Real *w_;          // step direction ( w_.size() == # active )
  Real *a_;   // correlation of columns of X with current step dir
  Real *L;  // lower triangular matrix of the gram matrix of X_A

  // temporaries
  Real *temp_;      // temporary storage for active correlations

  /** New Variable to exactly replicate matlab lars */
  bool stopcond;
  int vars;
  int nvars;
  int k;
  FILE* fid;


  inline Real sign(Real tmp) {
    if (tmp > 0) return 1.0;
    if (tmp < 0) return -1.0;
    return 0;
  }

  // get the current parameters
  const Idx* getParameter();
  // get least squares parameters for active set
  const void getParameters(Idx* p, const Idx* b);

  // constructor accepts a LarsDenseData object
  Lars(DenseLarsData *data):
    data_(data) {
    // initially all pareameters are 0, current residual = y
    // LILY: p is private???
    active_ = (int*) malloc(data_->p * sizeof(int));
    c_ = (Real*) calloc(data_->p * sizeof(Real));
    memset(active_, -1, sizeof(active_));
    nvars = min(data_->N - 1, data_->p);
    beta_ = (beta_pair*) malloc(nvars * sizeof(beta_pair));
    c_ = (Real*) calloc(nvars * sizeof(Real));
    w_ = (Real*) calloc(nvars * sizeof(Real));
    a_ = (Real*) calloc(nvars * sizeof(Real));
    L = (Real *) malloc(nvars * nvars * sizeof(Real));

    stopcond = 0;
    k = 0;
    vars = 0;
    //fid = fopen("vlarspp_debug.txt","w");
    fid = stderr;
    data_->getXtY(c_);
    // step dir = 0 so a_ = 0
    temp_ = (Real*) calloc(nvars * sizeof(Real));
  }

  ~Lars() {
    #ifdef DEBUG_PRINT
    fprintf(fid, "Lars() DONE\n");
    #endif
  }

  // Perform a single iteration of the LARS loop
  bool iterate() {
    if (vars >= nvars) return false;
    k++;
    #ifdef DEBUG_PRINT
    fprintf(fid, "K: %12d\n", k);
    fprintf(fid, "%12d %12d\n", vars, nvars);
    #endif

    Real C = 0.0;
    int j;
    for (int i = 0; i < data_->p; ++i) {
      if (active_[i] != -1) continue;
      if (fabs(c_[i]) > C) {
        j = i;
        C = fabs(c_[i]);
      }
      }
    #ifdef DEBUG_PRINT
    fprintf(fid, "[C,j] = [%12.5f, %12d]\n", C, j+1 );
    #endif
    #ifdef DEBUG_PRINT
    fprintf(fid, "activating %d\n", j+1 );
    #endif

    /**
     *  activate parameter j and updates cholesky
     * ------------------
     * Update state so that feature j is active with weight 0
     *  if it is not already active.
     *
     * R = cholinsert(R,X(:,j),X(:,A));
     * A = [A j];
     * I(I == j) = [];
     * vars = vars + 1;
     *
     **/
    //fprintf(fid, "activate(%d)\n", j );
    if ((active_[j] != -1) || vars >= data_->p) {
      fprintf(fid, "blash\n");
    } else {
      active_[j] = vars;
      beta_[vars](beta_pair(j, 0.0));

      for (int f = 0; f <= vars; ++f) {
        temp_[f] = data_->col_dot_product(j, beta_[f].first);
      }
      /// addRowCol, when vars==0, temp_ should be all zeros
      vars++;
      for (int i = 0; i < vars; ++i) {
        L[(vars - 1) * size + i] = temp_[i];
      }
      update_cholesky(L, vars, (vars - 1));

      // fprintf(fid, "vars %d\n", vars);
    }

    /**
     * computes w, AA, and X'w
     * -----------------------------
     * Solves for the step in parameter space given the current active parameters.
     **/
    Real AA = 0.0;
    // set w_ = sign(c_[A])
    for (int i = 0; i < vars; ++i) {
      w_[i] = sign(c_[beta_[i].first]);
    }
    //fprintf(fid, "sign(c_[A]):");
    //print(w_);
    // w_ = R\(R'\s)
    backsolve(L, w_, w_, vars);

    //fprintf(fid, "w_:");
    //print(w_);

    // AA = 1/sqrt(dot(GA1,s));
    for (int i = 0; i < vars; ++i) {
      AA += w_[i] * sign(c_[beta_[i].first]);
    }
    AA = 1.0 / sqrt(AA);
    //fprintf(fid, "AA: %12.5f\n", AA);
    for (int i = 0; i < vars; ++i) {
      w_[i] *= AA;
    }

    // calculate the a (uses beta to get active indices )
    // a_ = X'Xw
    data_->compute_direction_correlation(beta_, nvars, w_, nvars, a_ + 0);

    #ifdef DEBUG_PRINT
    fprintf(fid, "W:");
    //print(w_);
    fprintf(fid, "AA: %12.5f\n", AA);
    #endif

    Real gamma;
    if (vars == nvars) {
      gamma = C / AA;
      #ifdef DEBUG_PRINT
      fprintf(fid, "gamma: %12.5f\n", gamma);
      #endif
    } else {

      //fprintf(fid, "a:");
      //print(a_);
      gamma = C / AA;
      int min_index = -1;
      // temp = [(C - c(I))./(AA - a(I)); (C + c(I))./(AA + a(I))];
      // gamma = min([temp(temp > 0); C/AA]);
      for (int j = 0; j < vars; ++j) {
        // only consider inactive features
        if (active_[j] != -1) continue;
        Real t1 = (C - c_[j])/(AA - a_[j]);
        Real t2 = (C + c_[j])/(AA + a_[j]);
        // consider only positive items
        if (t1 > 0 && t1 < gamma) {
          gamma = t1;
          min_index = j;
        }
        if (t2 > 0 && t2 < gamma) {
          gamma = t2;
          min_index = j;
        }
      }

      #ifdef DEBUG_PRINT
      fprintf(fid, "min_index: %12d\n", min_index+1);
      fprintf(fid, "gamma: %12.5f\n", gamma);
      #endif
    }

    // add lambda * w to beta
    for(int i = 0; i < vars; ++i)
      beta_[i].second += gamma * w_[i];

    // update correlation with a
    for(int i = 0; i < vars; ++i)
      c_[i] -= gamma * a_[i];

    // print the beta
    #ifdef DEBUG_PRINT
    fprintf(fid, "beta: ");
    for (int i = 0; i < vars; ++i) {
      fprintf(fid, "%12.5f", beta_[i].second );
    }
    fprintf(fid, "\n");
    #endif

    return true;
  }
};


/** Return the Least-squares solution to X*beta = y for the subset
 * of currently active beta parameters */
void Lars::getParameters(beat_pair *p, const beta_pair *b) {
  Real *temp = (Real*) calloc(vars * sizeof(Real));
  Real *temp2 = (Real*) calloc(vars * sizeof(Real));

  data_->getXtY(temp);
  for (int i = 0; i < vars; ++i) {
    temp2[i] = temp[b[i].first];
  }

  backsolve(L, temp2, temp2, vars);

  for (int i = 0; i < vars; ++i) {
    p[i].first = b[i].first;
    p[i].second = temp2[i];
  }

  free(temp);
  free(temp2);
}

#endif
