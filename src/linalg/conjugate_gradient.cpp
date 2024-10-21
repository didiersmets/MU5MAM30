#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "conjugate_gradient.h"
#include "matrix.h"
#include "tiny_blas.h"

size_t conjugate_gradient_solve(const Matrix &A, const double *__restrict b,
				double *__restrict x, double *__restrict r,
				double *__restrict p, double *__restrict Ap,
				double tol, int max_iter)
{
	size_t N = A.rows;
	assert(A.rows == A.cols);

	/* Absolute squared tolerance */
	double normsqb = blas_dot(b, b, N) / N;
	double tol2 = (tol * tol) * normsqb;

	/* r_0 = b - Ax_0 */
	A.mvp(x, Ap);
	blas_copy(b, r, N);
	blas_axpy(-1, Ap, r, N);

	/* p_0 = r_0 */
	blas_copy(r, p, N);

	double error2 = blas_dot(r, r, N);

	int iter = 0;
	while (iter != max_iter && error2 > tol2) {
		iter++;
		/* alpha_k = r_k^Tr_k / (p_k^T A p_k) */
		A.mvp(p, Ap);
		double alpha = error2 / blas_dot(p, Ap, N);

		/* x_{k+1} = x_k + \alpha_k p_k */
		blas_axpy(alpha, p, x, N);
		/* r_{k+1} = r_k - \alpha_k Ap_k*/
		blas_axpy(-alpha, Ap, r, N);

		/* beta_k = r_{k+1}^Tr_{k+1} / (r_k^T r_k) */
		double beta = 1. / error2;
		error2 = blas_dot(r, r, N);
		/* TODO user callback instead */
		printf("Iteration %d : %g (%g)\r", iter, error2,
		       sqrt(error2 / normsqb));
		fflush(stdout);
		beta *= error2;

		/* p_{k+1} = r_{k+1} + beta_{k+1} p_k */
		blas_axpby(1, r, beta, p, N);
	}
	printf("\n");
	return iter;
}
