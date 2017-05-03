//-*-c++-*-
#ifndef LARS_INTERFACE_H
#define LARS_INTERFACE_H

#include "lars.h"
#include "dense_lars_data.h"

using namespace std;

/** FILE: lars_interface.h
 *
 * Contains set of interface routines for accessing the library's
 * high-level functionality.
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


/*-----------------------------------------------------------------------------
  Problem statement:

Variables:
- int N is the number of samples
- int p is the number of variables
- int r is the number of right-hand-sides
- T* X is (N x p) data matrix (N samples, p variables)
- T* Y is (N x r) response matrix in case of multiple right-hand-sides

- T lambda is the scalar L1-norm penalty weight
- T t is the L1-norm constraint

There are two equivalent ways to formulate the problem:

(1) L1-regularized formulation:

minimize norm(X*beta-y,2) + lambda*norm(beta,1)

(2) L1-constrained formulation:

minimize norm(X*beta-y,2)
subject to norm(beta,1) < t

In both cases, we use norm(~,n) to be the Ln-norm of ~.

There is a one-to-one mapping between lambda and t, though we never
actually solve for this mapping.

LARS and LASSO both provide many solutions to this problem, with various
values of lambda/t, correponding to various numbers of non-zero elements
in beta.

We provide several interfaces to this problem, described below.

---------------------------------------------------------------------------*/


namespace LARS{

  // Declare enum for stop type
  typedef enum {NONE=-1, NORM, LAMBDA, NUM_ITER, NUM_BETA} STOP_TYPE;

  //---------------------------------------------------------------------------
  // Interpolate SparseVector between beta_old and beta_new according to
  // interpolation_fraction
  //---------------------------------------------------------------------------
  template<typename T>
    inline void interpolateSparseVector(vector< pair<int,T> >* beta_interp,
        const int p,
        const vector< pair<int,T> >& beta_old,
        const vector< pair<int,T> >& beta_new,
        const T f) {
      T* beta_tmp = new T[2*p];
      flattenSparseVector(beta_old, beta_tmp, p);
      flattenSparseVector(beta_new, &beta_tmp[p], p);
      T tmp1, tmp2;
      for (int i=0;i<p;i++) {
        tmp1 = beta_tmp[i];
        tmp2 = beta_tmp[p+i];
        if (tmp1 != T(0) || tmp2 != T(0)) {
          beta_interp->push_back(pair<int,T>(i,(T(1)-f)*tmp1 + f*tmp2));
        }
      }
      delete [] beta_tmp;

    }
  //---------------------------------------------------------------------------
  // Computes L1-norm of SparseVector
  //---------------------------------------------------------------------------
  template<typename T>
    inline T l1NormSparseVector(const vector< pair<int,T> >& beta) {
      T norm = T(0);
      for(int i=0, n=beta.size(); i<n; ++i) {
        norm += fabs((T)beta[i].second);
      }
      return norm;
    }

  //---------------------------------------------------------------------------
  // Flattens a SparseVector (i.e., vector < pair<int,T> >)
  //---------------------------------------------------------------------------
  template<typename T>
    inline void flattenSparseVector(const vector< pair<int,T> >& beta,
        T* beta_dense, const int p) {
      memset(beta_dense,0,p*sizeof(T));
      for (int i=0,n=beta.size();i<n;i++) {
        beta_dense[beta[i].first] = beta[i].second;
      }
    }

  //---------------------------------------------------------------------------
  // Flattens a SparseMatrix (i.e., vector< vector < pair<int,T> > >)
  //---------------------------------------------------------------------------
  template<typename T>
    inline void flattenSparseMatrix(const vector< vector< pair<int,T> > >& beta,
        T* beta_dense, const int p, const int M) {
      memset(beta_dense,0,M*p*sizeof(T));
      for (int m=0;m<M;m++) {
        for (int i=0,n=beta[m].size();i<n;i++) {
          beta_dense[beta[m][i].first+m*p] = beta[m][i].second;
        }
      }
    }

  //---------------------------------------------------------------------------
  // Prints a C-array as a matrix with a label in Matlab style
  //---------------------------------------------------------------------------
  template<typename T>
    inline void printMatrix(T* a, const int N, const int p,
        std::string label, std::ostream& out) {
      out << label << " = [..." << endl;
      for(size_t r=0;r<N;r++) {
        for(size_t c=0;c<p;c++) {
          out.width(14);
          out << a[r+c*N] << " ";
        }
        out << endl;
      }
      out << "];" << endl;
    }

  //---------------------------------------------------------------------------
  // LARS Engine routine - accepts LarsData object
  //---------------------------------------------------------------------------
  template<typename T>
    inline int lars(vector< vector< pair<int,typename T::real> > >* beta,
        T& gld,
        const int N,
        const int p,
        const STOP_TYPE stop_type = -1,
        const typename T::real stop_val = T(0),
        const bool return_whole_path = true,
        const bool least_squares_beta = false,
        const bool verbose = false) {

      Lars<T> mylars(gld);
      typedef typename T::real U;

      // compute maximum number of beta vectors
      const int max_num_non_zero_beta = min(N,p);

      // internal storage for computing target value
      vector< pair<int,U> > beta_old;
      vector< pair<int,U> > beta_ls;
      vector< pair<int,U> > beta_interp;

      // internal variables
      bool target_reached = false;
      U interpolation_fraction = U(1);
      U lambda_old = (stop_type == LAMBDA)? gld.compute_lambda(beta_old): U(0);
      U norm_beta_old = U(0);
      U norm_beta_new, lambda_new;

      // do main LARS/LASSO loop
      for (int m=0;!target_reached;m++) {
        // perform iteration, returning on failure
        target_reached = false;
        if (!mylars.iterate()) {
          cerr << "LARS error: failed iteration.  M = "
            << mylars.getParameters().size()
            << endl << flush;
          target_reached = true;
          interpolation_fraction = U(1);
        }

        // grab local reference to current solution vector
        const vector< pair<int,U> >& beta_new = mylars.getParameters();
        int num_non_zero_beta = beta_new.size();

        // print iteration details
        //if (verbose) {

#ifdef DEBUG_PRINT
        cerr << " num_non_zero_beta: " << num_non_zero_beta << endl;
        //}
        cerr<<"l1norm:"<<l1NormSparseVector(beta_new)<<endl;
        cerr << "target_reached: " << target_reached << endl;
#endif
        // check stop conditions, if any given
        if ( target_reached == 0 ) {
#ifdef DEBUG_PRINT
          cerr << "NORM:" << NORM << endl;
          cerr << "stop_type" << stop_type << endl;
#endif

          switch (stop_type) {
            case NORM: // target norm of beta specified

              norm_beta_new = l1NormSparseVector(beta_new);
#ifdef DEBUG_PRINT
              cerr<<"norm_beta_new:"<<norm_beta_new<<endl;
#endif

              if (norm_beta_new >= stop_val) {
                interpolation_fraction = ((stop_val - norm_beta_old)/
                    (norm_beta_new - norm_beta_old));
//                cerr << "interpolation factor:" << interpolation_fraction << endl;
                target_reached = true;
              }
              break;

            case LAMBDA: // target lambda penalty specified
              lambda_new = gld.compute_lambda(beta_new);
              if (verbose) {
                cout << " lambda: " << lambda_new;
              }
              if (lambda_new <= stop_val) {
                interpolation_fraction = ((lambda_old - stop_val)/
                    (lambda_old - lambda_new));
                target_reached = true;
              }
              break;

            case NUM_ITER: // target number of iterations specified
              if (m >= int(stop_val)-1) {
                interpolation_fraction = U(1);
                target_reached = true;
              }
              break;

            case NUM_BETA: // target number of non-zero betas
              if (num_non_zero_beta >= int(stop_val)) {
                interpolation_fraction = U(1);
                target_reached = true;
              }
              break;
          }
        }

        if (verbose) {
          cout << ", frac: " << interpolation_fraction;
        }

        // check exit condition
        if (num_non_zero_beta >= max_num_non_zero_beta && !target_reached) {
          interpolation_fraction = U(1);
          target_reached = true;
        }

        if (target_reached) {
          if (least_squares_beta) {
            mylars.getParameters(&beta_ls, beta_new);
            beta->push_back(beta_ls);
          } else {
            if (interpolation_fraction == U(1)) {
              beta->push_back(beta_new);
            } else {
              interpolateSparseVector(&beta_interp, p, beta_old, beta_new,
                  interpolation_fraction);
              if (verbose) {
                cout << endl
                  << "final lambda: " << gld.compute_lambda(beta_interp);
              }
              beta->push_back(beta_interp);
            }
          }
        } else { // otherwise, store away current beta if that's required
          if (return_whole_path) {
            if (least_squares_beta) {
              mylars.getParameters(&beta_ls, beta_new);
              beta->push_back(beta_ls);
            }
            else beta->push_back(beta_new);
          }
        }

        // do end of loop stuff
        beta_old = beta_new;
        lambda_old = lambda_new;
        norm_beta_old = norm_beta_new;

        // print iteration details
        if (verbose) {
          cout << endl;
        }
      } // end main for loop

      // final steps
      if (verbose) {
        cout << endl
          << "LARS Output:" << endl
          << " - Number of returns: " << beta->size() << endl;
      }

      return beta->size();
    }
}; // end namespace LARS

#endif
