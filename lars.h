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

#include "dense_cholesky.h"

using namespace std;

template< typename T >
class Lars {
 public:
  typedef typename T::real real;

  inline real sign( real temp ) {
    if( temp > 0 ) return 1.0;
    if( temp < 0 ) return  -1.0;
    return 0;
  }
  /** get the current parameters */
  const vector<pair<int,real> >& getParameters();
  /** get least squares parameters for active set */
  const void getParameters(vector<pair<int,real> >* p,
		     const vector<pair<int,real> >& b);

  /** Constructor accepts a LarsDenseData object */
  Lars( T& data): data_(data), chol_( min(data.nrows(),data.ncols()))
  {
    // initially all parameters are 0, current residual = y
    nvars = min<int>(data_.nrows()-1,data_.ncols());
    active_ = (int*)malloc(data_.ncols(), sizeof(int));
    memset(active_, -1, sizeof(active_));
    c_ = (real*)calloc(nvars, sizeof(real));
    w_ = (real*)calloc(nvars, sizeof(real));
    a_ = (real*)calloc(nvars, sizeof(real));

    stopcond = 0;
    k = 0;
    vars = 0;
    //fid = fopen("vlarspp_debug.txt","w");
    fid = stderr;
    data_.getXtY( &c_ );
    // step dir = 0 so a_ = 0
    temp_ = (real*)calloc(nvars, sizeof(real));
  }

  ~Lars() {
#ifdef DEBUG_PRINT
    fprintf(fid, "DONE\n");
#endif

    //fclose(fid);
  }
  /** Perform a single interation of the LARS loop. */
  bool iterate(){
    if(vars >= nvars ) return false;
    k++;
#ifdef DEBUG_PRINT
    fprintf(fid, "K: %12d\n", k );
    fprintf(fid, "%12d %12d\n", vars, nvars);
#endif
    // [C j] = max(abs(c(I)));
    // j = I(j);
    real C = real(0); int j;
    for(int i=0; i<vars; ++i){
      if(active_[i] != -1) continue;
      if( fabs(c_[i]) > C ) {
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
    if((active_[j] != -1) || vars >= data_.nrows()) {
      fprintf(fid, "blash\n");
    }
    else {
      active_[j] = vars;
      beta_.push_back(make_pair(j,0.0));
      for(int f=0; f<vars; ++f){
        temp_[f] = data_.col_dot_product(j, beta_[f].first );
      }
      chol_.addRowCol( &temp_[0] );
      vars++;
      // fprintf(fid, "vars %d\n", vars );
    }

    /**
     * computes w, AA, and X'w
     * -----------------------------
     * Solves for the step in parameter space given the current active parameters.
     **/
    real AA(0.0);
    // set w_ = sign(c_[A])
    for(int i=0; i<vars; ++i){
      w_[i] = sign(c_[beta_[i].first]);
    }
    //fprintf(fid, "sign(c_[A]):");
    //print(w_);
    // w_ = R\(R'\s)
    chol_.solve(w_, &w_ );
    //fprintf(fid, "w_:");
    //print(w_);

    // AA = 1/sqrt(dot(GA1,s));
    for(int i=0; i<vars; ++i) AA += w_[i]*sign(c_[beta_[i].first]);
    AA = real(1.0)/sqrt(AA);
    //fprintf(fid, "AA: %12.5f\n", AA);
    for(int i=0; i<vars; ++i) w_[i] *= AA;


    // calculate the a (uses beta to get active indices )
    // a_ = X'Xw
    data_.compute_direction_correlation( beta_, w_, &(a_[0]) );

#ifdef DEBUG_PRINT
    fprintf(fid, "W:");
    //print(w_);
    fprintf(fid, "AA: %12.5f\n", AA);
#endif
    real gamma;
    if( vars == nvars ) {
      gamma = C/AA;
#ifdef DEBUG_PRINT
      fprintf(fid, "gamma: %12.5f\n", gamma);
#endif
    } else {
      //fprintf(fid, "a:");
      //print(a_);
      gamma = C/AA;
      int min_index = -1;
      // temp = [(C - c(I))./(AA - a(I)); (C + c(I))./(AA + a(I))];
      // gamma = min([temp(temp > 0); C/AA]);
      for(int j=0; j<vars; ++j) {
        // only consider inactive features
        if(active_[j] != -1) continue;
        real t1 = (C - c_[j])/(AA - a_[j]);
        real t2 = (C + c_[j])/(AA + a_[j]);
        // consider only positive items
        if( t1 > 0 && t1 < gamma ) {
          gamma = t1; min_index = j;
        }
        if( t2 > 0 && t2 < gamma ) {
          gamma = t2; min_index = j;
        }
      }
#ifdef DEBUG_PRINT
      fprintf(fid, "min_index: %12d\n", min_index+1);
      fprintf(fid, "gamma: %12.5f\n", gamma);
#endif
    }
    // add lambda * w to beta
    for(int i=0; i<vars; ++i)
      beta_[i].second += gamma * w_[i];

    // update correlation with a
    for(int i=0; i<vars; ++i)
      c_[i] -= gamma * a_[i];

    // print the beta
#ifdef DEBUG_PRINT
    fprintf(fid, "beta: ");
    for(int i=0; i<vars; ++i) {
      fprintf(fid, "%12.5f", beta_[i].second );
    }
    fprintf(fid, "\n");
#endif
    return true;
  }
  // member variables
  T& data_; // data(contains X and y)
  vector<pair<int,real> > beta_;   // current parameters(solution) [Not Sorted]

  // incrementally updated quantities
  int *active_ = (int*)calloc(nvars, sizeof(int)); // active[i] = position in beta of active param or -1
  real *c_ = (real*)calloc(nvars, sizeof(real)); // correlation of columns of X with current residual
  real *w_ = (real*)calloc(nvars, sizeof(real));          // step direction ( w_.size() == # active )
  real *a_ = (real*)calloc(nvars, sizeof(real));   // correlation of columns of X with current step dir
  DenseCholesky<real> chol_;   // keeps track of cholesky
  // temporaries
  real *temp_ = (real*)calloc(nvars, sizeof(real));      // temporary storage for active correlations


  /** New Variable to exactly replicate matlab lars */
  bool stopcond;
  int vars;
  int nvars;
  int k;
  FILE* fid;
};

/** Return a reference to the current active set of beta parameters. */
template<typename T>
const vector<pair<int, real> >& Lars<T>::getParameters() {
  return beta_;
}

/** Return the Least-squares solution to X*beta = y for the subset
 * of currently active beta parameters */
template<typename T>
const void Lars<T>::
getParameters(vector<pair<int,typename T::real> >* p,
	const vector<pair<int,typename T::real> >& b) {

  vector<real> temp(vars);
  vector<real> temp2(vars);

  p->resize(vars);
  data_.getXtY( &temp );
  for(int i=0; i<vars; ++i){
    temp2[i] = temp[b[i].first];
  }
  chol_.solve(temp2, &temp2 );
  for(int i=0; i<vars; ++i){
    (*p)[i].first=b[i].first;
    (*p)[i].second=temp2[i];
  }
}
#endif
