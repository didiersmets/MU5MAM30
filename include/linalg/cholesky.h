#include "sparse_matrix.h"

void in_place_cholesky_decomposition(SKLMatrix &A);

void cholesky_solve(const SKLMatrix &L, const double *__restrict b,
		    double *__restrict x, double *__restrict tmp);
