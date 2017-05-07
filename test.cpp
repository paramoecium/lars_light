#include <cstdio>

#include "util.h"
#include "lars.h"

template <class T>
inline T normalRand(T mean = T(0), T stdev = T(1)) {
  const double norm = 1.0/(RAND_MAX + 1.0);
  double u = 1.0 - std::rand()*norm;
  double v = rand()*norm;
  double z = sqrt(-2.0*log(u))*cos(2.0*M_PI*v);
  return T(mean + stdev*z);
}

template <class T>
inline void prepareData(const int D, const int K, const int r,
			                  const bool norm, T *X, T *y) {
  //X = new T[D*K];
  //y = new T[D*r];
  for (int j = 0, k = 0; j < K; j++) {
    T s = T(0);
    T s2 = T(0);
    for (int i=0;i<D;i++,k++) {
      T v = normalRand<T>();
      X[k] = v;
      s += v;
      s2 += v*v;
    }
    if (norm) {
      T std = sqrt(s2 - s*s/T(D));
      k -= D;
      for (int i=0;i<D;i++,k++) {
         X[k] = (X[k] - s/T(D))/std;
      }
    }
  }

  for (int i=0;i<D*r;i++) {
    y[i] = normalRand<T>();
  }
}

int main() {

  // Initailize data
  int D, K;
  Real *Xt;
  Real *y;
  Idx *beta;
  Real lambda = 0.1;

  D = 6, K = 3;
  Xt = (Real*) malloc(D * K * sizeof(Real));
  y = (Real*) malloc(D * sizeof(Real));
  beta = (Idx*) malloc(K * sizeof(Idx));

  prepareData(D, K, 1, true, Xt, y);
  /*
  Xt[0 * D + 0] = 1;
  Xt[0 * D + 1] = 2;
  Xt[0 * D + 2] = 1;
  Xt[1 * D + 0] = 1;
  Xt[1 * D + 1] = 1;
  Xt[1 * D + 2] = 0;
  Xt[2 * D + 0] = 1;
  Xt[2 * D + 1] = 0;
  Xt[2 * D + 2] = 1;


  y[0] = 5;
  y[1] = 2;
  y[2] = 4;
  */

  Lars lars(Xt, y, D, K, lambda);

  lars.solve();

  lars.getParameters(&beta);
  printf("get Parameters\n");

  for (int i = 0; i < lars.active_itr; i++)
    printf("%d : %.3f\n", beta[i].id, beta[i].v);

  for (int i = 0; i < lars.active_itr; i++) {
    for (int j = 0; j < D; j++)
      y[j] -= Xt[beta[i].id * D + j] * beta[i].v;
  }
  Real sqr_error = Real(0.0);
  for (int j = 0; j < D; j++)
    sqr_error += y[j] * y[j];
  printf("error = %.3f\n", sqrt(sqr_error));
}
