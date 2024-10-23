#include "matrix.h"

size_t conjugate_gradient_solve(const Matrix &A, const double *__restrict b,
				double *__restrict x, double *__restrict r,
				double *__restrict p, double *__restrict Ap,
				double *__restrict rel_error, double tol,
				int max_iter, bool inited = false);
