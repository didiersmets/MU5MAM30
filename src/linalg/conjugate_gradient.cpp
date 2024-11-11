#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "conjugate_gradient.h"
#include "matrix.h"
#include "tiny_blas.h"

double cg_iterate_once(const Matrix &A, double *__restrict x,
		       double *__restrict r, double *__restrict p,
		       double *__restrict Ap, double r2)
{
	size_t N = A.rows;
	assert(A.rows == A.cols);

	/* alpha_k = r_k^Tr_k / (p_k^T A p_k) */
	A.mvp(p, Ap);
	double alpha = r2 / blas_dot(p, Ap, N);

	/* x_{k+1} = x_k + \alpha_k p_k */
	blas_axpy(alpha, p, x, N);
	/* r_{k+1} = r_k - \alpha_k Ap_k*/
	blas_axpy(-alpha, Ap, r, N);

	/* r2_new = r_{k+1}^T r_{k+1} */
	double r2_new = blas_dot(r, r, N);

	/* beta_k = r_{k+1}^Tr_{k+1} / (r_k^T r_k) */
	/* p_{k+1} = r_{k+1} + beta_{k+1} p_k */
	double beta = r2_new / r2;
	blas_axpby(1, r, beta, p, N);

	return (r2_new);
}

size_t conjugate_gradient_solve(const Matrix &A, const double *__restrict b,
				double *__restrict x, double *__restrict r,
				double *__restrict p, double *__restrict Ap,
				double *rel_error, double tol, int max_iter,
				bool inited)
{
	size_t N = A.rows;
	assert(A.rows == A.cols);

	double b2 = blas_dot(b, b, N);

	if (!inited) {
		/* r_0 = b - Ax_0 */
		A.mvp(x, r);
		blas_axpby(1, b, -1, r, N);
		/* p_0 = r_0 */
		blas_copy(r, p, N);
	}

	double r2 = blas_dot(r, r, N);
	*rel_error = sqrt(r2 / b2);

	int iter = 0;
	while ((iter < max_iter) && (*rel_error > tol)) {
		r2 = cg_iterate_once(A, x, r, p, Ap, r2);
		*rel_error = sqrt(r2 / b2);
		iter++;
	}
	return iter;
}
