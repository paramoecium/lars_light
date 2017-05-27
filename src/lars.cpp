#include "lars.h"
#include "timer_id.h"
#include <algorithm>
#include <immintrin.h>

#ifndef LARS_CPP
#define LARS_CPP

Lars::Lars(const Real *Xt_in, int D_in, int K_in, Real lambda_in, Timer &timer_in):
    Xt(Xt_in), D(D_in), K(K_in), lambda(lambda_in), timer(timer_in) {

  beta_id = (int*) calloc(K, sizeof(int));
  beta_old_id = (int*) calloc(K, sizeof(int));
  beta_v  = (Real*) calloc(K, sizeof(Real));
  beta_old_v  = (Real*) calloc(K, sizeof(Real));

  // Initializing
  active_size = fmin(K, D);
  active = (Real*) malloc(K * sizeof(Real));

  c = (Real*) calloc(K, sizeof(Real));
	sgn = (Real*) calloc(active_size, sizeof(Real));
  w = (Real*) calloc(active_size, sizeof(Real));
  L = (Real*) calloc(active_size * active_size, sizeof(Real));
  G = (Real*) calloc(active_size * active_size, sizeof(Real));
  u = (Real*) calloc(D, sizeof(Real));
  a = (Real*) calloc(K, sizeof(Real));
	tmp_int = (int *) calloc(active_size, sizeof(int));
  tmp = (Real*) calloc((K>D?K:D), sizeof(Real));

	gamma = 0.0;
}

void Lars::set_y(const Real *y_in) {
  y = y_in;

  memset(beta_id, 0, K*sizeof(int));
  memset(beta_old_id, 0, K*sizeof(int));
  memset(beta_v, 0, K*sizeof(Real));
  memset(beta_old_v, 0, K*sizeof(Real));

  active_itr = 0;
  memset(w, 0, active_size * sizeof(Real));
  memset(u, 0, D * sizeof(Real));
  memset(a, 0, K * sizeof(Real));
  memset(L, 0, active_size * active_size * sizeof(Real));

	for (int i = 0; i < active_size; i++) tmp_int[i] = i;

  for (int i = 0; i < K; i++) {
    active[i] = -1;
    for (int j = 0; j < D; j++) {
      c[i] += Xt[i * D + j] * y[j];
    }
  }
}

Lars::~Lars() {
  print("Lars() DONE\n");
}

bool Lars::iterate() {
  if (active_itr >= active_size) return false;

  Real C = 0.0;
  int cur = -1;
  const __m256d zero = _mm256_setzero_pd();
  const __m256d mone = _mm256_set1_pd(-1);
  const __m256d pone = _mm256_set1_pd(1);

  timer.start(GET_ACTIVE_IDX);
  __m256d maxv = _mm256_setzero_pd();
  __m256d curv = _mm256_set1_pd(-1);
  __m256d posv = _mm256_set_pd(-1.0, -2.0, -3.0, -4.0);
  __m256d gamma_v = _mm256_set1_pd(-gamma);
  for (int i = 0; i < K; i+=4) {
    __m256d active_v = _mm256_load_pd(&active[i]);
    __m256d cc = _mm256_load_pd(&c[i]);
    __m256d a_v = _mm256_load_pd(&a[i]);
    cc = _mm256_fmadd_pd(gamma_v, a_v, cc);
    _mm256_store_pd(&c[i], cc);
    if (active[i+0]>=0) sgn[(int)active[i+0]] = sign(c[i+0]);
    if (active[i+1]>=0) sgn[(int)active[i+1]] = sign(c[i+1]);
    if (active[i+2]>=0) sgn[(int)active[i+2]] = sign(c[i+2]);
    if (active[i+3]>=0) sgn[(int)active[i+3]] = sign(c[i+3]);
//    c_v = _mm256_fmadd_pd(gamma_v, a_v, c_v);

    active_v = _mm256_cmp_pd(active_v, zero, _CMP_LT_OS);
    __m256d neg_cc = _mm256_cmp_pd(cc, zero, _CMP_LT_OS);
    __m256d active_cc = _mm256_and_pd(cc, active_v);
    __m256d ccx2 = _mm256_mul_pd(active_cc, _mm256_set1_pd(-2.0));
    neg_cc = _mm256_and_pd(ccx2, neg_cc);
    __m256d fabs_cc = _mm256_add_pd(active_cc, neg_cc);
    //_mm256_store_pd(tmp, fabs_cc);

    __m256d change = _mm256_cmp_pd(maxv, fabs_cc, _CMP_LT_OS);
    maxv = _mm256_max_pd(maxv, fabs_cc);
    posv = _mm256_add_pd(posv, _mm256_set1_pd(4.0));
    curv = _mm256_blendv_pd(curv, posv, change);

  }
  _mm256_store_pd(tmp, maxv);
  _mm256_store_pd(tmp + 4, curv);
  for (int i = 0; i < 4; i++) {
    if (tmp[i] > C) {cur = (int)tmp[i+4]; C = tmp[i];} 
  }


//  timer.start(GET_ACTIVE_IDX);
//	Real C = 0.0;
//	int cur = -1;
//  __m256d gamma_v = _mm256_set1_pd(-gamma);
//  __m256d max_c = _mm256_setzero_pd();
//  __m256d cur_v = _mm256_set1_pd(-1.0);
//  __m256d pos_v = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
//  __m256d four = _mm256_set1_pd(4.0);
//	for (int i = 0; i < K; i+=4) {
//    __m256d a_v = _mm256_load_pd(&a[i]);
//    __m256d c_v = _mm256_load_pd(&c[i]);
//    __m256d active_v = _mm256_load_pd(&active[i]);
//
//    c_v = _mm256_fmadd_pd(gamma_v, a_v, c_v);
//
//    __m256d not_active = _mm256_cmp_pd(active_v, zero, _CMP_LT_OS);
//    _mm256_store_pd(tmp, not_active);
//    __m256d neg_sgn_c = _mm256_cmp_pd(c_v, zero, _CMP_LT_OS);
//    __m256d pos_sgn_c = _mm256_cmp_pd(zero, c_v, _CMP_LT_OS);
//
//    __m256d pos_c = _mm256_and_pd(c_v, pos_sgn_c);
//    __m256d abs_c = _mm256_fmsub_pd(_mm256_set1_pd(2.0), pos_c, c_v);
//    __m256d use_c = _mm256_cmp_pd(max_c, abs_c, _CMP_LT_OS); // C < fabs(c[i])
//    use_c = _mm256_and_pd(not_active, use_c); // active[i] < 0 and fabs(c[i]) > C
//    __m256d sgn_c = _mm256_add_pd(_mm256_and_pd(mone, neg_sgn_c),
//                                  _mm256_and_pd(pone, pos_sgn_c));
//
//    cur_v = _mm256_blendv_pd(cur_v, pos_v, use_c);
//    max_c = _mm256_blendv_pd(max_c, abs_c, use_c);
//
//    pos_v = _mm256_add_pd(four, pos_v);
//
//    _mm256_store_pd(tmp, sgn_c);
//    _mm256_store_pd(&c[i], c_v);
//
//    if(active[i+0] >= 0) sgn[(int)active[i+0]] = tmp[0];
//    if(active[i+1] >= 0) sgn[(int)active[i+1]] = tmp[1];
//    if(active[i+2] >= 0) sgn[(int)active[i+2]] = tmp[2];
//    if(active[i+3] >= 0) sgn[(int)active[i+3]] = tmp[3];
//	}
//  
//  _mm256_store_pd(tmp, max_c);
//  _mm256_store_pd(tmp+4, cur_v);

//  for (int i = 0; i < 4; i++) {
//    if (C < tmp[i]) cur = tmp[i+4], C = tmp[i];
//  }


//	for (int i = 0; i < K; ++i) {
//		c[i] -= gamma * a[i];
//
//		if (active[i] < 0 and fabs(c[i]) > C) {
//			cur = i;
//			C = fabs(c[i]);
//		} else if (active[i] >= 0) {
//			sgn[(int)active[i]] = sign(c[i]);
//		}
//	}
  timer.end(GET_ACTIVE_IDX);

  // All remainging C are 0
  if (cur == -1) return false;

  assert((int)active[cur] == -1 and active_itr < D);

  active[cur] = active_itr;
  beta_id[active_itr] = cur;
  beta_v[active_itr] = 0;
	sgn[active_itr] = sign(c[cur]);


  //TODO: Fuse here
  // calculate Xt_A * Xcur, Matrix * vector
  // new active row to add to gram matrix of active set

  timer.start(FUSED_CHOLESKY);
  Real AA = update_cholesky_n_solve(L, w, sgn, active_itr, active_size, Xt, cur, beta_id, beta_v, D);
  timer.end(FUSED_CHOLESKY);

  // get the actual w[]
  // get a = X' X_a w
  // Now do X' (X_a w)

  memset(a, 0, K*sizeof(Real));
  memset(u, 0, D*sizeof(Real));
	std::sort(tmp_int, tmp_int + (active_itr + 1), [this](int i, int j) {return beta_id[i]<beta_id[j];});
  // u = X_a * w
  for (int i = 0; i <= active_itr; ++i) {
		w[tmp_int[i]] *= AA;
    for (int j = 0; j < D; j++) {
      u[j] += w[tmp_int[i]] * Xt[beta_id[tmp_int[i]] * D + j];
    }
  }

  // a = X' * u
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < D; j++) {
      a[i] += Xt[i * D + j] * u[j];
    }
  }

  gamma = C / AA;
  int gamma_id = cur;
  if (active_itr < active_size) {
    print("C=%.3f AA=%.3f\n", C, AA);
    for (int i = 0; i < K; i++) {
      if ((int)active[i] >= 0) continue;
      Real t1 = (C - c[i]) / (AA - a[i]);
      Real t2 = (C + c[i]) / (AA + a[i]);
      print("%d : t1 = %.3f, t2 = %.3f\n", i, t1, t2);

      if (t1 > 0 and t1 < gamma) gamma = t1, gamma_id=i;
      if (t2 > 0 and t2 < gamma) gamma = t2, gamma_id=i;
    }
  }
  print("gamma = %.3f from %d col\n", gamma, gamma_id);

  // add gamma * w to beta
  for (int i = 0; i <= active_itr; ++i)
    beta_v[i] += gamma * w[i];


  active_itr++;

  return true;
}

void Lars::solve() {
  int itr = 0;
  while (iterate()) {
    // compute lambda_new
    print("=========== The %d Iteration ends ===========\n\n", itr);

    //calculateParameters();
    lambda_new = compute_lambda();

    print("---------- lambda_new : %.3f lambda_old: %.3f lambda: %.3f\n", lambda_new, lambda_old, lambda);
    for (int i = 0; i < active_itr; i++)
      print("%d : %.3f %.3f\n", beta_id[i], beta_v[i], beta_old_v[i]);

    if (lambda_new > lambda) {
      lambda_old = lambda_new;
      memcpy(beta_old_v, beta_v, active_itr * sizeof(Real));
      beta_old_v[active_itr-1] = beta_v[active_itr-1];
    } else {
      Real factor = (lambda_old - lambda) / (lambda_old - lambda_new);
      for (int j = 0; j < active_itr; j++) {
        beta_v[j] = beta_old_v[j] + factor * (beta_v[j] - beta_old_v[j]);
      }
      break;
    }

    itr++;

  }
  print("LARS DONE\n");
}

void Lars::getParameters(int** beta_out_id, Real** beta_out_v) const {
  *beta_out_id = beta_id;
  *beta_out_v = beta_v;
}

void Lars::getParameters(Real* beta_out) const {
  memset(beta_out, 0, K * sizeof(Real));
  for (int i = 0; i < active_itr; i++) {
    beta_out[beta_id[i]] = beta_v[i];
  }
}


// computes lambda given beta, lambda = max(abs(2*X'*(X*beta - y)))
inline Real Lars::compute_lambda() {
  // compute (y - X*beta)
  memcpy(tmp, y, D * sizeof(Real));
  for (int i = 0; i < active_itr; i++) {
    for (int j = 0; j < D; j++)
      tmp[j] -= Xt[beta_id[tmp_int[i]] * D + j] * beta_v[tmp_int[i]];
  }
  // compute X'*(y - X*beta)
  Real max_lambda = Real(0.0);
  for (int i = 0; i < K; ++i) {
    Real lambda = 0;
    for (int j = 0; j < D; j++) lambda += Xt[i * D + j] * tmp[j];
    max_lambda = fmax(max_lambda, fabs(lambda));
  }
  return max_lambda;
}


#endif
