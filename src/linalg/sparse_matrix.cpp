#include "sparse_matrix.h"

double CSRMatrix::operator()(int i, int j) const
{
	assert(i >= 0 && i < (int)rows);
	size_t start = K[i];
	size_t end = K[i + 1];
	for (size_t k = start; k < end; ++k) {
		if (J[k] == j)
			return AIJ[k];
	}
	return 0;
}

bool CSRMatrix::set_at(int i, int j, double aij)
{
	assert(i >= 0 && i < (int)rows);
	size_t start = K[i];
	size_t end = K[i + 1];
	for (size_t k = start; k < end; ++k) {
		if (J[k] == j) {
			AIJ[k] = aij;
			return true;
		}
	}
	return false;
}

void CSRMatrix::mvp(const double *__restrict x, double *__restrict y) const
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

double CSRMatrix::sum() const
{
	double res = 0.0;
	for (size_t k = 0; k < nnz; k++) {
		res += AIJ[k];
	}
	return res;
}

