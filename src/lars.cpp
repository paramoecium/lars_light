#include "lars.h"
#include "timer_id.h"
#include <algorithm>
#include <immintrin.h>

#ifndef LARS_CPP
#define LARS_CPP
#define G(i, j) G[((i * (i + 1))>>1) - 1 + j]

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
//  L = (Real*) calloc(((active_size * (active_size + 1))>>1) - 1 + active_size, sizeof(Real));
//  G = (Real*) calloc(((active_size * (active_size + 1))>>1) - 1 + active_size, sizeof(Real));
  L = (Real*) calloc(active_size * active_size / 2 + active_size * 2, sizeof(Real));
  G = (Real*) calloc(active_size * active_size / 2 + active_size * 2, sizeof(Real));
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

  timer.start(INIT_CORRELATION);
  for (int i = 0; i < K; i += 4) {
    active[i+0] = -1;
    active[i+1] = -1;
    active[i+2] = -1;
    active[i+3] = -1;
    __m256d cc0 = _mm256_setzero_pd();
    __m256d cc1 = _mm256_setzero_pd();
    __m256d cc2 = _mm256_setzero_pd();
    __m256d cc3 = _mm256_setzero_pd();

    for (int x = 0; x < D; x += 4) {
      __m256d yy = _mm256_load_pd(&y[x]);
      __m256d xt0 = _mm256_load_pd(&Xt[(i+0) * D + x]);
      __m256d xt1 = _mm256_load_pd(&Xt[(i+1) * D + x]);
      __m256d xt2 = _mm256_load_pd(&Xt[(i+2) * D + x]);
      __m256d xt3 = _mm256_load_pd(&Xt[(i+3) * D + x]);
      cc0 = _mm256_add_pd(cc0, _mm256_mul_pd(xt0, yy));
      cc1 = _mm256_add_pd(cc1, _mm256_mul_pd(xt1, yy));
      cc2 = _mm256_add_pd(cc2, _mm256_mul_pd(xt2, yy));
      cc3 = _mm256_add_pd(cc3, _mm256_mul_pd(xt3, yy));
    }
    __m256d cc01 = _mm256_hadd_pd(cc0, cc1);
    __m256d cc23 = _mm256_hadd_pd(cc2, cc3);
    _mm256_store_pd(tmp, cc01);
    _mm256_store_pd(tmp+4, cc23);
    c[i] = tmp[0] + tmp[2];
    c[i+1] = tmp[1] + tmp[3];
    c[i+2] = tmp[4] + tmp[6];
    c[i+3] = tmp[5] + tmp[7];
  }
  // mvm(Xt, false, y, c, K, D);
  timer.end(INIT_CORRELATION);
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

    active_v = _mm256_cmp_pd(active_v, zero, _CMP_LT_OS);
    __m256d neg_cc = _mm256_cmp_pd(cc, zero, _CMP_LT_OS);
    __m256d active_cc = _mm256_and_pd(cc, active_v);
    __m256d ccx2 = _mm256_mul_pd(active_cc, _mm256_set1_pd(-2.0));
    neg_cc = _mm256_and_pd(ccx2, neg_cc);
    __m256d fabs_cc = _mm256_add_pd(active_cc, neg_cc);

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
  Real AA = update_cholesky_n_solve(L, G, w, sgn, active_itr, active_size, Xt, cur, beta_id, beta_v, D);
  timer.end(FUSED_CHOLESKY);

  // get the actual w[]
  // get a = X' X_a w
  // Now do X' (X_a w)

  memset(a, 0, K*sizeof(Real));
  memset(u, 0, D*sizeof(Real));
	std::sort(tmp_int, tmp_int + (active_itr + 1), [this](int i, int j) {return beta_id[i]<beta_id[j];});
  // u = X_a * w
  // a = X' * u
	timer.start(GET_A);
	memset(tmp, 0, (1+active_itr)*sizeof(Real));
	for (int i = 0; i <= active_itr; i++) {
		w[i] *= AA;
		for (int j = 0; j < i; j++) {
			tmp[j] += G(i, j) * w[i];
			tmp[i] += G(i, j) * w[j];
		}
		tmp[i] += G(i, i) * w[i];
	} 

	for (int i = 0; i <= active_itr; ++i) {
		for (int j = 0; j < D; j++) {
			u[j] += w[i] * Xt[beta_id[i] * D + j];
		}
	}
	
	for (int i = 0; i < K; i++) {
		if (active[i] >= 0) a[i] = tmp[(int)active[i]];
		else {
			for (int j = 0; j < D; j++) {
				a[i] += Xt[i * D + j] * u[j];
			}
		}
	}

	timer.end(GET_A);

  timer.start(GET_GAMMA);
  gamma = C / AA;
  int gamma_id = cur;
  if (active_itr < active_size) {
    print("C=%.3f AA=%.3f\n", C, AA);
    __m256d cc = _mm256_set1_pd(C);
    __m256d aa = _mm256_set1_pd(AA);
    __m256d min_p = _mm256_set1_pd(gamma);
    __m256d min_m = _mm256_set1_pd(gamma);
    for (int i = 0; i < K; i+=4) {
      __m256d c_v = _mm256_load_pd(&c[i]);
      __m256d a_v = _mm256_load_pd(&a[i]);
      __m256d c_m = _mm256_sub_pd(cc, c_v);
      __m256d a_m = _mm256_sub_pd(aa, a_v);
      __m256d g_m = _mm256_div_pd(c_m, a_m);
      __m256d c_p = _mm256_add_pd(cc, c_v);
      __m256d a_p = _mm256_add_pd(aa, a_v);
      __m256d g_p = _mm256_div_pd(c_p, a_p);
//      print256("g_p", g_p);print256("g_m",g_m);
//      for (int k = 0; k < 4; k++)     printf("%.3f ", (C + c[i+k]) / (AA + a[i+k])); printf("\n");
//      for (int k = 0; k < 4; k++)     printf("%.3f ", (C - c[i+k]) / (AA - a[i+k])); printf("\n"); printf("\n");

      __m256d not_active = _mm256_load_pd(&active[i]);
      not_active = _mm256_cmp_pd(not_active, zero, _CMP_LT_OS);

      __m256d take_p = _mm256_cmp_pd(zero, g_p, _CMP_LT_OS);
      __m256d small_p= _mm256_cmp_pd(g_p, min_p, _CMP_LT_OS);
      __m256d ok_p = _mm256_and_pd(take_p, not_active);
      min_p = _mm256_blendv_pd(min_p, g_p, _mm256_and_pd(ok_p, small_p));
      
      __m256d take_m = _mm256_cmp_pd(zero, g_m, _CMP_LT_OS);
      __m256d small_m= _mm256_cmp_pd(g_m, min_m, _CMP_LT_OS);
      __m256d ok_m = _mm256_and_pd(take_m, not_active);
      min_m = _mm256_blendv_pd(min_m, g_m, _mm256_and_pd(ok_m, small_m));
    }
    _mm256_store_pd(tmp, min_m);
    _mm256_store_pd(tmp+4, min_p);
    for (int i = 0; i < 8; i++) if (tmp[i] > 0 and tmp[i] < gamma) gamma = tmp[i];
  }
  timer.end(GET_GAMMA);


  int V_size = (active_itr + 1) / 16, V_res = (active_itr+1)%16;
  __m256d g_gw = _mm256_set1_pd(gamma);
  for (int ii = 0; ii < V_size; ++ii) {
    int i = ii * 16;
    __m256d bv0 = _mm256_load_pd(&beta_v[i]);
    __m256d ww0 = _mm256_load_pd(&w[i]);
    __m256d bv1 = _mm256_load_pd(&beta_v[i+4]);
    __m256d ww1 = _mm256_load_pd(&w[i+4]);
    __m256d bv2 = _mm256_load_pd(&beta_v[i+8]);
    __m256d ww2 = _mm256_load_pd(&w[i+8]);
    __m256d bv3 = _mm256_load_pd(&beta_v[i+12]);
    __m256d ww3 = _mm256_load_pd(&w[i+12]);
    bv0 = _mm256_fmadd_pd(g_gw, ww0, bv0);
    bv1 = _mm256_fmadd_pd(g_gw, ww1, bv1);
    bv2 = _mm256_fmadd_pd(g_gw, ww2, bv2);
    bv3 = _mm256_fmadd_pd(g_gw, ww3, bv3);
    _mm256_store_pd(&beta_v[i], bv0);
    _mm256_store_pd(&beta_v[i+4], bv1);
    _mm256_store_pd(&beta_v[i+8], bv2);
    _mm256_store_pd(&beta_v[i+12], bv3);
  }
  for (int ii = 0; ii < V_res; ++ii) {
    int i = 16 * V_size + ii;
    beta_v[i] += gamma * w[i];
  }

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
  timer.start(COMPUTE_LAMBDA);
  memcpy(tmp, y, D * sizeof(Real));
  memcpy(u, y, D * sizeof(Real));
//  for (int i = 0; i < active_itr; i++) {
//    for (int j = 0; j < D; j++)
//      tmp[j] -= Xt[beta_id[tmp_int[i]] * D + j] 
//                * beta_v[tmp_int[i]];
//  }
//  // compute X'*(y - X_A*beta)
//  Real max_lambda = Real(0.0);
//  for (int i = 0; i < K; ++i) {
//    Real lambda = 0;
//    for (int j = 0; j < D; j++) 
//      lambda += Xt[i * D + j] * tmp[j];
//    max_lambda = fmax(max_lambda, fabs(lambda));
//  }

  Real max_lambda = 1e-5;
//	int B_size = 32, B_cnt = active_itr / B_size;
//	for (int b_i = 0; b_i < B_cnt; b_i += B_size) {
//		for (int b_j = 0; b_j < b_i; b_j += B_size) {
//			for (int j = b_j; j < b_j + B_size; j++) {
//				for (int i = b_i; i < b_i + B_size; i++) {
//					tmp[j] -= G(j, i) * beta_v[i];
//					tmp[i] -= G(j, i) * beta_v[j];
//				}
//			}
//		}
//
//		// b_j == b_i
//		for (int j = b_i; j < b_i + B_size; j++) {
//			tmp[j] -= G(j, j) * beta_v[j];
//			for (int i = b_i; i < b_i + B_size; i++) {
//				tmp[j] -= G(j, i) * beta_v[i];
//				tmp[i] -= G(j, i) * beta_v[j];
//			}
//		}
//	}
//	for (int j = B_cnt * B_size; j < active_itr; j++) {
//		for (int i = 0; i < j; i++) {
//			tmp[i] -= G(j, i) * beta_v[j];
//			tmp[j] -= G(j, i) * beta_v[i];
//			max_lambda = fmax(max_lambda, tmp[i]);
//		}
//		tmp[j] -= G(j, j) * beta_v[j];
//		max_lambda = fmax(max_lambda, tmp[j]);
//	}
  for (int i = 0; i < active_itr; i++) {
    for (int j = 0; j < i; j++) {
      tmp[j] -= G(i, j) * beta_v[i];
      if (i == active_itr - 1) max_lambda = fmax(max_lambda, tmp[j]);
      tmp[i] -= G(i, j) * beta_v[j];
    }
    tmp[i] -= G(i, i) * beta_v[i];
  }
  max_lambda = fmax(max_lambda, tmp[active_itr-1]);


  timer.end(COMPUTE_LAMBDA);
  return max_lambda;
}



#endif
