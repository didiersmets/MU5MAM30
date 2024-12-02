#include "math.h"

#include "sparse_matrix.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

void in_place_cholesky_decomposition(SKLMatrix &A)
{
	for (uint32_t i = 0; i < A.rows; ++i) {
		uint32_t ioff = A.jmin[i];
		double *__restrict Li = &A.data[A.row_start[i]] - ioff;
		for (uint32_t j = 0; j < i; ++j) {
			uint32_t joff = A.jmin[j];
			const double *__restrict Lj =
				&A.data[A.row_start[j]] - joff;
			uint32_t jstart = MAX(ioff, joff);
			double sum = 0.0;
			for (uint32_t k = jstart; k < j; ++k) {
				sum += Li[k] * Lj[k];
			}
			Li[j] = (Li[j] - sum) / Lj[j];
		}
		double sum = 0.0;
		for (uint32_t j = ioff; j < i; ++j) {
			sum += Li[j] * Li[j];
		}
		assert(Li[i] > sum);
		Li[i] = sqrt(Li[i] - sum);
	}
}

void cholesky_solve(const SKLMatrix &L, const double *__restrict b,
		    double *__restrict x, double *__restrict tmp)
{
	L.fwd_substitution(tmp, b);
	L.bwd_substitution(x, tmp);
}
