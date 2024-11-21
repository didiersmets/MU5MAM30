#include "sparse_matrix.h"

void CRSMatrix::mvp(const double *__restrict x, double *__restrict y) const
{
	for (size_t i = 0; i < rows; ++i) {
		y[i] = 0;
		for (int k = K[i]; k < K[i + 1]; ++k) {
			assert((size_t)k < nnz);
			assert((size_t)J[k] < cols);
			y[i] += AIJ[k] * x[J[k]];
		}
	}
}

double CRSMatrix::sum() const
{
	double res = 0.0;
	for (size_t k = 0; k < nnz; k++) {
		res += AIJ[k];
	}
	return res;
}
