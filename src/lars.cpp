#include <immintrin.h>

#include "lars.h"
#include "timer_id.h"

#ifndef LARS_CPP
#define LARS_CPP

Lars::Lars(const Real *Xt_in, int D_in, int K_in, Real lambda_in, Timer &timer_in):
    Xt(Xt_in), D(D_in), K(K_in), lambda(lambda_in), timer(timer_in) {

  beta = (Idx*) calloc(K, sizeof(Idx));
  beta_old = (Idx*) calloc(K, sizeof(Idx));

  // Initializing
  active_size = fmin(K, D);
  active = (int*) malloc(K * sizeof(int));

  c = (Real*) calloc(K, sizeof(Real));
  w = (Real*) calloc(active_size, sizeof(Real));
  sgn = (Real*) calloc(active_size, sizeof(Real));
  u = (Real*) calloc(D, sizeof(Real));
  a = (Real*) calloc(K, sizeof(Real));
  L = (Real*) calloc(active_size * active_size, sizeof(Real));
  tmp = (Real*) calloc((K>D?K:D), sizeof(Real));
}

void Lars::set_y(const Real *y_in) {
  y = y_in;

  memset(beta, 0, K*sizeof(Idx));
  memset(beta_old, 0, K*sizeof(Idx));

  active_itr = 0;
  memset(active, -1, K * sizeof(int));
  memset(w, 0, active_size * sizeof(Real));
  memset(u, 0, D * sizeof(Real));
  memset(a, 0, K * sizeof(Real));
  memset(L, 0, active_size * active_size * sizeof(Real));

  timer.start(INIT_CORRELATION);
  mvm(Xt, false, y, c, K, D);
  timer.end(INIT_CORRELATION);
}

Lars::~Lars() {
  print("Lars() DONE\n");
}

bool Lars::iterate() {
  if (active_itr > active_size) return false;

  Real C = 0.0;
  int cur = -1;
  timer.start(GET_ACTIVE_IDX);
  for (int i = 0; i < K; ++i) {
    print("c[%d]=%.3f active[i]=%d\n", i, c[i], active[i]);
    if (active[i] != -1) continue;
    if (fabs(c[i]) > C) {
      cur = i;
      C = fabs(c[i]);
    }
  }
  timer.end(GET_ACTIVE_IDX);


  // All remainging C are 0
  if (cur == -1) return false;

  print("Active set size is now %d\n", active_itr + 1);
  print("Activate %d column with %.3f value\n", cur, C);

  print("active[%d]=%d cur=%d D=%d\n", active_itr, active[cur], cur, D);
  assert(active[cur] == -1 and active_itr < D);

  active[cur] = active_itr;
  beta[active_itr] = Idx(cur, 0);


  // calculate Xt_A * Xcur, Matrix * vector
  // new active row to add to gram matrix of active set
  timer.start(UPDATE_GRAM_MATRIX);
  for (int i = 0; i <= active_itr; ++i) {
    __m256 sum0 = _mm256_setzero_pd();
    __m256 sum1 = _mm256_setzero_pd();
    for (int x = 0; x < D; x += 8) {
      __m256 cur0 = _mm256_load_pd(&Xt[cur * D + x]);
      __m256 xt0  = _mm256_load_pd(&Xt[beta[i].id * D + x]);
      sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(cur0, xt0));

      __m256 cur1 = _mm256_load_pd(&Xt[cur * D + x + 4]);
      __m256 xt1  = _mm256_load_pd(&Xt[beta[i].id * D + x + 4]);
      sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(cur1, xt1));
    }
    sum0 = _mm256_hadd_pd(sum0, sum1);
    sum0 = _mm256_hadd_pd(sum0, sum0);
    _mm256_store_pd(tmp, sum0);
    L[active_itr*active_size + i] = tmp[0] + tmp[2];
  }

//  for (int i = 0; i <= active_itr; ++i) {
//    L[active_itr*active_size + i] = dot(Xt + cur * D, Xt + beta[i].id * D, D);
//  }
  timer.end(UPDATE_GRAM_MATRIX);
  

  timer.start(UPDATE_CHOLESKY);
  update_cholesky(L, active_itr, active_size);
  timer.end(UPDATE_CHOLESKY);


  // set w[] = sign(c[])
  timer.start(INITIALIZE_W);
  for (int i = 0; i <= active_itr; ++i) {
    sgn[i] = sign(c[beta[i].id]);
  }
  timer.end(INITIALIZE_W);


  // w = R\(R'\s)
  // w is now storing sum of all rows? in the inverse of G_A
  timer.start(BACKSOLVE_CHOLESKY);
  backsolve(L, w, sgn, active_itr+1, active_size);
  timer.end(BACKSOLVE_CHOLESKY);


  // AA is is used to finalize w[]
  // AA = 1 / sqrt(sum of all entries in the inverse of G_A);
  timer.start(GET_AA);
  Real AA = 0.0;
  int i;
  for (i = 0; i <= active_itr-4; i+=4) {
    __m256 ww = _mm256_load_pd(&w[i]);
    __m256 sn = _mm256_load_pd(&sgn[i]);
    __m256 aa = _mm256_mul_pd(ww, sn);
    __m256 ha = _mm256_hadd_ps(aa, ww); 
    Real a4[4];
    _mm256_store_pd(a4, ha);
    AA += (a4[0] + a4[2]);
  }
  //TODO: Remove residual 
  for (; i <=active_itr; i++) {
    AA += w[i] * sgn[i];
  }
//  for (int i = 0; i <= active_itr; ++i) {
//    AA += w[i] * sign(c[beta[i].id]);
//  }
  AA = 1.0 / sqrt(AA);
  print("AA: %.3f\n", AA);
  timer.end(GET_AA);


  // get the actual w[]
  timer.start(GET_W);
  for (i = 0; i <= active_itr-4; i+=4) {
    __m256 ww = _mm256_load_pd(&w[i]);
    __m256 aa = _mm256_set1_pd(AA);
    _mm256_store_pd(&w[i], _mm256_mul_pd(ww, aa));
  }
  //TODO : Remove residual
  for (; i <= active_itr; i++) {
    w[i] *= AA;
  }
  //for (int i = 0; i <= active_itr; ++i) {
  //  w[i] *= AA;
  //}
  print("w solved :");
  for (int i = 0; i < D; ++i) print("%.3f ", w[i]);
  print("\n");
  timer.end(GET_W);

  // get a = X' X_a w
  // Now do X' (X_a w)
  // can store a X' X_a that update only some spaces when adding new col to
  // activation set
  // Will X'(X_a w) be better? // more
  // X' (X_a w) more flops less space?
  // (X' X_a) w less flops more space?
  memset(a, 0, K*sizeof(Real));
  memset(u, 0, D*sizeof(Real));
  // u = X_a * w
  timer.start(GET_U);
  for (int i = 0; i <= active_itr; ++i) {
    axpy(w[i], &Xt[beta[i].id * D], u, D);
  }
  timer.end(GET_U);


  // a = X' * u
  timer.start(GET_A);
  mvm(Xt, false, u, a, K, D);
  timer.end(GET_A);


  print("u : ");
  for (int i = 0; i < D; i++) print("%.3f ", u[i]);
  print("\n");

  print("a : ");
  for (int i = 0; i < K; i++) print("%.3f ", a[i]);
  print("\n");


  timer.start(GET_GAMMA);
  Real gamma = C / AA;
  int gamma_id = cur;
  if (active_itr < active_size) {
    print("C=%.3f AA=%.3f\n", C, AA);
    for (int i = 0; i < K; i++) {
      if (active[i] != -1) continue;
      Real t1 = (C - c[i]) / (AA - a[i]);
      Real t2 = (C + c[i]) / (AA + a[i]);
      print("%d : t1 = %.3f, t2 = %.3f\n", i, t1, t2);

      if (t1 > 0 and t1 < gamma) gamma = t1, gamma_id=i;
      if (t2 > 0 and t2 < gamma) gamma = t2, gamma_id=i;
    }
  }
  print("gamma = %.3f from %d col\n", gamma, gamma_id);
  timer.end(GET_GAMMA);

  // add gamma * w to beta
  timer.start(UPDATE_BETA);
  for (int i = 0; i <= active_itr; ++i)
    beta[i].v += gamma * w[i];
  timer.end(UPDATE_BETA);

  // update correlation with a
  timer.start(UPDATE_CORRELATION);
  for (int i = 0; i < K; ++i)
    c[i] -= gamma * a[i];
  timer.end(UPDATE_CORRELATION);

  print("beta: ");
  for (int i = 0; i <= active_itr; ++i) print("%d %.3f ", beta[i].id, beta[i].v);
  print("\n");

  active_itr++;
  return true;
}

void Lars::solve() {
  int itr = 0;
  while (iterate()) {
    // compute lambda_new
    print("=========== The %d Iteration ends ===========\n\n", itr);

    //calculateParameters();
    timer.start(GET_LAMBDA);
    lambda_new = compute_lambda();
    timer.end(GET_LAMBDA);

    print("---------- lambda_new : %.3f lambda_old: %.3f lambda: %.3f\n", lambda_new, lambda_old, lambda);
    for (int i = 0; i < active_itr; i++)
      print("%d : %.3f %.3f\n", beta[i].id, beta[i].v, beta_old[i].v);

    if (lambda_new > lambda) {
      lambda_old = lambda_new;
      memcpy(beta_old, beta, active_itr * sizeof(Idx));
    } else {
      Real factor = (lambda_old - lambda) / (lambda_old - lambda_new);
      timer.start(INTERPOLATE_BETA);
      for (int j = 0; j < active_itr; j++) {
//        beta[j].v = beta_old[j].v * (1.f - factor) + factor * beta[j].v;
        beta[j].v = beta_old[j].v + factor * (beta[j].v - beta_old[j].v);
      }
      timer.end(INTERPOLATE_BETA);
      break;
    }
    itr++;
  }
  print("LARS DONE\n");
}


//void Lars::calculateParameters() {
//  for (int i = 0; i < active_itr; i++) {
//    print("beta[%d] = %.3f\n", beta[i].id, beta[i].v);
//  }
//
//  mvm(Xt, false, y, tmp, D, K);
//
//  Real *tmp2 = (Real*) calloc(active_itr, sizeof(Real));
//  for (int i = 0; i < active_itr; ++i) {
//    tmp2[i] = tmp[beta[i].id];
//  }
//  backsolve(L, tmp2, tmp2, active_itr, active_size);
//
//  for (int i = 0; i < active_itr; ++i) {
//    beta[i].id = beta[i].id;
//    beta[i].v = tmp2[i];
//  }
//
//  free(tmp2);
//}

void Lars::getParameters(Idx** beta_out) const {
  *beta_out = beta;
}

void Lars::getParameters(Real* beta_out) const {
  memset(beta_out, 0, K * sizeof(Real));
  for (int i = 0; i < active_itr; i++) {
    beta_out[beta[i].id] = beta[i].v;
  }
}


// computes lambda given beta, lambda = max(abs(2*X'*(X*beta - y)))
inline Real Lars::compute_lambda() {
  // compute (y - X*beta)
  memcpy(tmp, y, D * sizeof(Real));
  for (int i = 0; i < active_itr; i++) {
    for (int j = 0; j < D; j++)
      tmp[j] -= Xt[beta[i].id * D + j] * beta[i].v;
  }
  // compute X'*(y - X*beta)
  Real max_lambda = Real(0.0);
  for (int i = 0; i < K; ++i) {
    max_lambda = fmax(max_lambda, fabs(dot(Xt + i * D, tmp, D)));
  }
  return max_lambda;
}

#endif
