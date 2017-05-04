// line 161
// cblas_ddot...
dot(&X[N*c1], &X[N*c2], N)

// line 172
//cblas_dgemv(CblasColMajor
//cols = p
//rows = N
//NOTE: X is column major and transposed
//NOTE: transpose (X,X_new)
matVecProd(double * X, double * y, double * Xty, int p, int N)

// line 180
//cblas_dgemm(CblasColMajor
//cols = p
//rows = N
gramMatrix(double * X, double * XtX_buf, double * rows, double * cols)

// line 197
//cblas_daxpy()
daxpy(wval[i], XtX_col[beta[i].first], a, p)

// line 203
//cblas_daxpy()
daxpy(wval[i], &X[beta[i].first*N], Xw, p)

// line 205
// now do X'*(X*w)
// cblas_dgemv(CblasColMajor,CblasTrans,N,p,1.0,X,N,Xw,1,0.0,a,1);
matVecProd(Xprime, Xw, rows_of_Xprime, cols_of_Xprime)

// line 222
//cblas_daxpy(p, -beta[i].second, XtX_col[beta[i].first], 1, tmp_p, 1)
daxpy(-beta[i].second, XtX_facecol[beta[i].first], tmp_p, p);

// line 229
//cblas_daxpy(N, -beta[i].second, &X[N*beta[i].first], 1, Xw, 1);
daxpy(-beta[i].second, &X[N*beta[i].first], Xw, N);

// line 231
// now compute 2*X'*Xw = 2*X'*(y - X*beta)
//cblas_dgemv(CblasColMajor,CblasTrans,N,p,2.0,X,N,Xw,1,0.0,tmp_p,1);
// NOTE how to implement this?
scalarMultiplyVector(2,X,X2,N);
// NOTE transposition of matrix of alternative implementation?
//transpose(X,Xprime, rows, cols);
matVecProd(X, true, X2, N, p);

// file dense_cholesky:
// line 92/95
//cblas_ddot( N, a, 1, b, 1);
dot(a, b, N);