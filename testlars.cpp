//-*-c++-*-

/** Test code to evaluate a random LARS problem.
 *
 *  - User supplies N, p as command-line args (in none given, defaults used).
 *  - User supplies optional number of right-hand-sides.
 *  - Generates random problem data and runs Lars on it, reporting results.
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

#include <cstdlib>
#include <time.h>
#include <math.h>
#include <iostream>
#include <sys/time.h>

#include "lars_interface.h"

using namespace std;

#define M_PI       3.14159265358979323846
/// Returns a sample from a normal distribution
template <class T>
inline T normalRand(T mean = T(0), T stdev = T(1)) {
  const double norm = 1.0/(RAND_MAX + 1.0);
  double u = 1.0 - std::rand()*norm;
  double v = rand()*norm;
  double z = sqrt(-2.0*log(u))*cos(2.0*M_PI*v);
  return T(mean + stdev*z);
}

/// Generate random problem data (X and y) of size (Nxp), optionally
/// normalized so that each column is zero mean and unit variance.
template <class T>
inline void prepareData(const int N, const int p, const int r,
			const bool norm,
			T*& X, T*& y) {
  X = new T[N*p];
  y = new T[N*r];
  for (int j=0,k=0;j<p;j++) {
    T s = T(0);
    T s2 = T(0);
    for (int i=0;i<N;i++,k++) {
      T v = normalRand<T>();
      X[k] = v;
      s += v;
      s2 += v*v;
    }
    if (norm) {
      T std = sqrt(s2 - s*s/T(N));
      k -= N;
      for (int i=0;i<N;i++,k++) {
	X[k] = (X[k] - s/T(N))/std;
      }
    }
  }

  for (int i=0;i<N*r;i++) {
    y[i] = normalRand<T>();
  }
}


class Timer {
public:
  // Constructor/Destructor
  Timer();

  // Methods
  void start(); // starts watch
  double stop(); // stops watch, returning time
  void reset(); // reset watch to zero
  double print(char* label = NULL); // prints time on watch without stopping
  double stopAndPrint(char* label = NULL); // stops and prints time

private:
  // Data
  bool timing;
  long double start_time;
  long double stop_time;
  long double total_time;
};

inline Timer::Timer() {
  reset();
}

inline void Timer::start() {
  if (!timing) {
    start_time = (long double)(clock())/(long double)(CLOCKS_PER_SEC);
    timing = true;
  }
}

inline double Timer::stop() {
  if (timing) {
    stop_time = (long double)(clock())/(long double)(CLOCKS_PER_SEC);
    timing = false;
    total_time += (stop_time - start_time);
  }
  return (double)total_time;
}

inline void Timer::reset() {
  total_time = start_time = stop_time = 0.0;
  timing = false;
}

inline double Timer::print(char* label) {
  bool was_timing = timing;
  double current_time = stop();
  if (label) cout << label;
  cout.precision(4);
  cout << current_time << " seconds" << endl << flush;
  if (was_timing) start();
  return current_time;
}

inline double Timer::stopAndPrint(char* label) {
  double current_time = stop();
  if (label) cout << label;
  cout.precision(4);
  cout << current_time << " seconds" << endl << flush;
  return current_time;
}

///////////////////////////////////////////////////////////////////////////
////////////////////////// User-Editable Portion //////////////////////////
///////////////////////////////////////////////////////////////////////////

// We support both double and float.  Change this typedef to select which.
typedef double real;

// Define default problem size.
const int p_default = 64;
const int N_default = 64;

// For the case of solving for multiple right-hand-sides
const int r = 1;

// To have accurate times, we allow several trials for each size.
const int num_rhs_default = 1;

// Define which type of problem to test (0 for LARS, 1 for LASSO).
const LARS::METHOD method = LARS::LARS;
const LARS::STOP_TYPE stop_type = LARS::NORM;
const real stop_val = real(1);
const bool return_whole_path = true;
const bool least_squares_beta = false;
const bool verbose = true;
const bool use_multiple_right_hand_sides = false;
const KERNEL kernel = AUTO;
const bool precomputed = false;

///////////////////////////////////////////////////////////////////////////
//////////////////////// End User-Editable Portion /////////////////////////
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]){
  // Give message about calling conventions
  if (argc <= 1) {
    cout << endl
	 << "Usage: " << endl
	 << "testlars [N] [p] [# RHS]" << endl
	 << " - Data matrix is (N x p) " << endl
	 << " - # RHS is the number of right-hand side vectors to consider."
	 << endl << endl;
  }

  // Data for dense lars problem
  real* X;
  real* Y;
  real* norm_beta;
  vector< vector< pair<int,real> > > beta;

  // Grab command line args
  int p = p_default;
  int N = N_default;
  int num_rhs = num_rhs_default;

  if (argc > 1) N = atoi(argv[1]);
  if (argc > 2) p = atoi(argv[2]);
  if (argc > 3) num_rhs = atoi(argv[3]);

  // Generate random data (function prepareData(...) is in miscutil.h
  //seedRand();
  srand(1);
  prepareData(N, p, r, false, X, Y);

  cout << "Testing LARS Library on random data: "
       << "X is " << N << " x " << p << ", "
       << "Y is " << N << " x " << r << "..." << endl << flush;

  // Run LARS
  int M;
  Timer t;
  t.start();

  for (int i=0;i<num_rhs;i++) {
    beta.clear();
    if (use_multiple_right_hand_sides) {
      M = LARS::lars(&beta, X, Y, N, p, r, stop_type, stop_val,
		     least_squares_beta, verbose, kernel, precomputed);
    } else {
      M = LARS::lars(&beta, X, Y, N, p, stop_type, stop_val,
		     return_whole_path, least_squares_beta, verbose,
		     kernel, precomputed);
    }
  }

  if (M <= 0) {
    cerr << "Bad return value. Exiting early. " << endl << flush;
    exit(1);
  }
  cout << "Done." << endl << endl;
  cout << "M = " << M << endl;
  t.stopAndPrint("Total time: " );

  // Print if small enough
  if (max(p,N) < 20) {
    // Make array of beta norms
    norm_beta = new real[beta.size()];
    for (int i=0;i<beta.size();i++) {
      norm_beta[i] = LARS::l1NormSparseVector(beta[i]);
    }

    // Make beta dense for printing
    real* beta_dense = new real[M*p];
    LARS::flattenSparseMatrix(beta,beta_dense,p,M);

    // print results
    cout << endl;
    cout.precision(8);
    LARS::printMatrix(X,N,p,"X",cout);
    LARS::printMatrix(Y,N,r,"Y",cout);
    LARS::printMatrix(beta_dense,p,M,"b",cout);
    LARS::printMatrix(norm_beta,1,beta.size(),"norm_beta",cout);

    // clean up
    delete [] beta_dense;
    delete [] norm_beta;
  }

  // clean up
  delete [] X;
  delete [] Y;
}
