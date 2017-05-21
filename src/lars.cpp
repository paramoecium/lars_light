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
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    for (int x = 0; x < D; x += 8) {
      __m256d cur0 = _mm256_load_pd(&Xt[cur * D + x]);
      __m256d xt0  = _mm256_load_pd(&Xt[beta[i].id * D + x]);
      sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(cur0, xt0));

      __m256d cur1 = _mm256_load_pd(&Xt[cur * D + x + 4]);
      __m256d xt1  = _mm256_load_pd(&Xt[beta[i].id * D + x + 4]);
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


  int V_size = (1 + active_itr)/4, V_res = (1 + active_itr)%4;

  // AA is is used to finalize w[]
  // AA = 1 / sqrt(sum of all entries in the inverse of G_A);
  timer.start(GET_AA);
  Real AA = 0.0;
  __m256d aa = _mm256_setzero_pd();
  for (int ii = 0; ii < V_size; ++ii) {
    int i = ii * 4;
    __m256d ww = _mm256_load_pd(&w[i]);
    __m256d sg = _mm256_load_pd(&sgn[i]);
    aa = _mm256_add_pd(aa, _mm256_mul_pd(ww, sg));
  }
  aa = _mm256_hadd_pd(aa, aa);
  _mm256_store_pd(tmp, aa);
  AA = (tmp[0] + tmp[2]);
  for (int ii = 0; ii < V_res; ii++) {
    int i = 4*V_size + ii;
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

  for (int ii = 0; ii < V_size; ++ii) {
    int i = ii * 4;
    __m256d ww = _mm256_load_pd(&w[i]);
    __m256d aa = _mm256_set1_pd(AA);
    _mm256_store_pd(&w[i], _mm256_mul_pd(ww, aa));
  }
  //TODO : Remove residual
  for (int ii = 0; ii < V_res; ii++) {
    int i = 4 * V_size + ii;
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
  // TODO: Merge GET_U and GET_A ?
  // TODO: Unroll to 2 to compensate load store time
  timer.start(GET_U);
  for (int i = 0; i <= active_itr; ++i) {
    __m256d ww = _mm256_set1_pd(w[i]);
    for (int x = 0; x < D; x += 4) {
      __m256d uu = _mm256_load_pd(&u[x]);
      __m256d xa = _mm256_load_pd(&Xt[beta[i].id * D + x]);
      __m256d xa_w = _mm256_mul_pd(ww, xa);
      uu = _mm256_add_pd(uu, xa_w);
      _mm256_store_pd(&u[x], uu);
    }
  }
//  for (int i = 0; i <= active_itr; ++i) {
//    axpy(w[i], &Xt[beta[i].id * D], u, D);
//  }
  timer.end(GET_U);


  // a = X' * u
  timer.start(GET_A);
  //mvm(Xt, false, u, a, K, D);
  for (int y = 0; y < K; y++) {
    __m256d sum = _mm256_setzero_pd();
    for (int x = 0; x < D; x += 4) {
      __m256d uu = _mm256_load_pd(&u[x]);
      __m256d xt = _mm256_load_pd(&Xt[y * D + x]);
      __m256d xu = _mm256_mul_pd(uu, xt);
      sum = _mm256_add_pd(sum, xu);
    }
    sum = _mm256_hadd_pd(sum, sum);
    _mm256_store_pd(tmp, sum);
    a[y] = tmp[0] + tmp[2];
  }
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
    __m256d cc_c = _mm256_set1_pd(C);
    __m256d aa_c  = _mm256_set1_pd(AA);
    for (int i = 0; i < K; i+=4) {
      __m256d cc = _mm256_load_pd(&c[i]);
      __m256d aa = _mm256_load_pd(&a[i]);
      __m256d c_minus = _mm256_add_pd(cc_c, -cc);
      __m256d c_plus  = _mm256_add_pd(cc_c, cc);
      __m256d a_minus = _mm256_add_pd(aa_c, -aa);
      __m256d a_plus  = _mm256_add_pd(aa_c, aa);
      __m256d ca_minus = _mm256_div_pd(c_minus, a_minus);
      __m256d ca_plus  = _mm256_div_pd(c_plus, a_plus);

      _mm256_store_pd(tmp, ca_minus);
      _mm256_store_pd(tmp+4, ca_plus);
      int gamma0_id = cur, gamma1_id = cur;
      Real gamma0 = gamma, gamma1 = gamma;
      for (int j = 0; j < 4; j++) {
        if (active[i+j] != -1) continue;
        if (tmp[j] > 0 and tmp[j] < gamma0) 
            gamma0 = tmp[j], gamma0_id = i+j;
        if (tmp[j+4] > 0 and tmp[j+4] < gamma1) 
            gamma1 = tmp[j+4], gamma1_id = i+j;
      }
      if (gamma0 < gamma1) gamma = gamma0, gamma_id = gamma0_id;
      else                 gamma = gamma1, gamma_id = gamma1_id;

    }
  }
//  if (active_itr < active_size) {
//    print("C=%.3f AA=%.3f\n", C, AA);
//    for (int i = 0; i < K; i++) {
//      if (active[i] != -1) continue;
//      Real t1 = (C - c[i]) / (AA - a[i]);
//      Real t2 = (C + c[i]) / (AA + a[i]);
//      print("%d : t1 = %.3f, t2 = %.3f\n", i, t1, t2);
//
//      if (t1 > 0 and t1 < gamma) gamma = t1, gamma_id=i;
//      if (t2 > 0 and t2 < gamma) gamma = t2, gamma_id=i;
//    }
//  }
  print("gamma = %.3f from %d col\n", gamma, gamma_id);
  timer.end(GET_GAMMA);

  // add gamma * w to beta
  // TODO: separate struct Idx to 2 array
  timer.start(UPDATE_BETA);
  __m256d g_gw = _mm256_set1_pd(gamma);
  for (int ii = 0; ii < V_size; ++ii) {
    int i = ii * 4;
    __m256d ww = _mm256_load_pd(&w[i]);
    __m256d gw = _mm256_mul_pd(g_gw, ww);
    _mm256_store_pd(tmp, gw);
    for (int j = 0; j < 4; j++) beta[i+j].v += tmp[j];
  }
  for (int ii = 0; ii < V_res; ++ii) {
    int i = 4 * V_size + ii;
    beta[i].v += gamma * w[i];
  }
//  for (int i = 0; i <= active_itr; ++i)
//    beta[i].v += gamma * w[i];
  timer.end(UPDATE_BETA);

  // update correlation with a
  timer.start(UPDATE_CORRELATION);
  __m256d g_ga = _mm256_set1_pd(gamma);
  for (int i = 0; i < K; i+= 4) {
    __m256d cc = _mm256_load_pd(&c[i]);
    __m256d aa = _mm256_load_pd(&a[i]);
    __m256d ga = _mm256_mul_pd(g_ga, aa);
    _mm256_store_pd(&c[i], _mm256_add_pd(cc, -ga));
  }
//  for (int i = 0; i < K; ++i) 
//    c[i] -= gamma * a[i];
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
    __m256d beta_v = _mm256_set1_pd(beta[i].v);
    int beta_id = beta[i].id;
    for (int j = 0; j < D; j+=4) {
      __m256d yy = _mm256_load_pd(&tmp[j]);
      __m256d xt = _mm256_load_pd(&Xt[beta_id + j]);
      __m256d y_xtb = _mm256_add_pd(yy, -_mm256_mul_pd(xt, beta_v));
      _mm256_store_pd(&tmp[j], y_xtb);
    }
  }
  //memcpy(tmp, y, D * sizeof(Real));
  //for (int i = 0; i < active_itr; i++) {
  //  for (int j = 0; j < D; j++)
  //    tmp[j] -= Xt[beta[i].id * D + j] * beta[i].v;
  //}
  // compute X'*(y - X*beta)
  Real max_lambda = Real(0.0);
  for (int i = 0; i < K; ++i) {
    Real lambda_tmp[5] = {0};
    for (int j = 0; j < D; j+=8) {
      __m256d xt0 = _mm256_load_pd(&Xt[i * D + j + 0]);
      __m256d y_xtb0 = _mm256_load_pd(&tmp[j + 0]);
      __m256d lambda0 = _mm256_mul_pd(xt0, y_xtb0);

      __m256d xt1 = _mm256_load_pd(&Xt[i * D + j + 4]);
      __m256d y_xtb1 = _mm256_load_pd(&tmp[j + 4]);
      __m256d lambda1 = _mm256_mul_pd(xt1, y_xtb1);

      __m256d lambda01 = _mm256_hadd_pd(lambda0, lambda1);
      lambda01 = _mm256_hadd_pd(lambda01, lambda01);
      _mm256_store_pd(lambda_tmp, lambda01);
      lambda_tmp[4] += (lambda_tmp[0] + lambda_tmp[2]);
    }
    max_lambda = fmax(max_lambda, fabs(lambda_tmp[4]));
  }
//  Real max_lambda = Real(0.0);
//  for (int i = 0; i < K; ++i) {
//    max_lambda = fmax(max_lambda, fabs(dot(Xt + i * D, tmp, D)));
//  }
  return max_lambda;
}

#endif
