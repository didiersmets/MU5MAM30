#include "sparse_matrix.h"

double &CSRMatrix::operator()(uint32_t i, uint32_t j)
{
	static double dummy = 0.0;
	assert(i < rows);
	size_t start = row_start[i];
	size_t stop = row_start[i + 1];
	for (size_t k = start; k < stop; ++k) {
		if (col[k] == j)
			return data[k];
	}
	/* Not a valid matrix entry */
	assert(false);
	return dummy;
}

void CSRMatrix::mvp(const double *__restrict x, double *__restrict y) const
{
	for (size_t i = 0; i < rows; ++i) {
		y[i] = 0;
		size_t start = row_start[i];
		size_t stop = row_start[i + 1];
		for (uint32_t k = start; k < stop; ++k) {
			assert(k < nnz);
			assert(col[k] < cols);
			y[i] += data[k] * x[col[k]];
		}
	}
	if (symmetric) {
		for (size_t i = 0; i < rows; ++i) {
			size_t start = row_start[i];
			/* stop before the diagonal */
			size_t stop = row_start[i + 1] - 1;
			for (uint32_t k = start; k < stop; ++k) {
				y[col[k]] += data[k] * x[i];
			}
		}
	}
}

double CSRMatrix::sum() const
{
	double res = 0.0;
	for (size_t k = 0; k < nnz; k++) {
		res += data[k];
	}
	if (symmetric) {
		res *= 2;
		for (size_t k = 0; k < rows; k++) {
			assert(col[row_start[k + 1] - 1] == k);
			res -= data[row_start[k + 1] - 1];
		}
	}
	return res;
}
