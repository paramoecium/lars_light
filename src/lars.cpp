#include "lars.h"
#include "timer_id.h"
#include <algorithm>

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
  active = (int*) malloc(K * sizeof(int));

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
  memset(active, -1, K * sizeof(int));
  memset(w, 0, active_size * sizeof(Real));
  memset(u, 0, D * sizeof(Real));
  memset(a, 0, K * sizeof(Real));
  memset(L, 0, active_size * active_size * sizeof(Real));

	for (int i = 0; i < active_size; i++) tmp_int[i] = i;

  timer.start(INIT_CORRELATION);
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < D; j++) {
      c[i] += Xt[i * D + j] * y[j];
    }
  }
  timer.end(INIT_CORRELATION);
}

Lars::~Lars() {
  print("Lars() DONE\n");
}

bool Lars::iterate() {
  if (active_itr >= active_size) return false;

	Real C = 0.0;
	int cur = -1;
	timer.start(GET_ACTIVE_IDX);
	for (int i = 0; i < K; ++i) {
		c[i] -= gamma * a[i];

		if (active[i] < 0 and fabs(c[i]) > C) {
			cur = i;
			C = fabs(c[i]);
		} else if (active[i] >= 0) {
			sgn[active[i]] = sign(c[i]);
		}
	}
	timer.end(GET_ACTIVE_IDX);

  // All remainging C are 0
  if (cur == -1) return false;

  assert(active[cur] == -1 and active_itr < D);

  active[cur] = active_itr;
  beta_id[active_itr] = cur;
  beta_v[active_itr] = 0;
	sgn[active_itr] = sign(c[cur]);


  //TODO: Fuse here
  // calculate Xt_A * Xcur, Matrix * vector
  // new active row to add to gram matrix of active set
  timer.start(UPDATE_GRAM_MATRIX);
  for (int i = 0; i <= active_itr; ++i) {
    L[active_itr * active_size + i] = 0;
    for (int j = 0; j < D; j++) {
      L[active_itr * active_size + i] += Xt[cur * D + j] * Xt[beta_id[i] * D + j];
    }
  }
  timer.end(UPDATE_GRAM_MATRIX);


  timer.start(UPDATE_CHOLESKY);
  update_cholesky(L, active_itr, active_size);
  timer.end(UPDATE_CHOLESKY);


  // w = R\(R'\s)
  // w is now storing sum of all rows? in the inverse of G_A
  timer.start(BACKSOLVE_CHOLESKY);
  backsolve(L, w, sgn, active_itr+1, active_size);
  timer.end(BACKSOLVE_CHOLESKY);


  // AA is is used to finalize w[]
  // AA = 1 / sqrt(sum of all entries in the inverse of G_A);
  timer.start(GET_AA);
  Real AA = 0.0;
  for (int i = 0; i <= active_itr; ++i) {
    AA += w[i] * sgn[i];
  }
  AA = 1.0 / sqrt(AA);
  print("AA: %.3f\n", AA);
  timer.end(GET_AA);


  // get the actual w[]
  // get a = X' X_a w
  // Now do X' (X_a w)

  memset(a, 0, K*sizeof(Real));
  memset(u, 0, D*sizeof(Real));
	std::sort(tmp_int, tmp_int + (active_itr + 1), [this](int i, int j) {return beta_id[i]<beta_id[j];});
  // u = X_a * w
  timer.start(GET_U);// Fuse GET_W
  for (int i = 0; i <= active_itr; ++i) {
		w[tmp_int[i]] *= AA;
    for (int j = 0; j < D; j++) {
      u[j] += w[tmp_int[i]] * Xt[beta_id[tmp_int[i]] * D + j];
    }
  }
  timer.end(GET_U);

  // a = X' * u
  timer.start(GET_A);
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < D; j++) {
      a[i] += Xt[i * D + j] * u[j];
    }
  }
  timer.end(GET_A);

  timer.start(GET_GAMMA);
  gamma = C / AA;
  int gamma_id = cur;
  if (active_itr < active_size) {
    print("C=%.3f AA=%.3f\n", C, AA);
    for (int i = 0; i < K; i++) {
      if (active[i] >= 0) continue;
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
    beta_v[i] += gamma * w[i];
  timer.end(UPDATE_BETA);


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
      print("%d : %.3f %.3f\n", beta_id[i], beta_v[i], beta_old_v[i]);

    if (lambda_new > lambda) {
      lambda_old = lambda_new;
      memcpy(beta_old_v, beta_v, active_itr * sizeof(Real));
      beta_old_v[active_itr-1] = beta_v[active_itr-1];
    } else {
      Real factor = (lambda_old - lambda) / (lambda_old - lambda_new);
      timer.start(INTERPOLATE_BETA);
      for (int j = 0; j < active_itr; j++) {
        beta_v[j] = beta_old_v[j] + factor * (beta_v[j] - beta_old_v[j]);
      }
      timer.end(INTERPOLATE_BETA);
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
