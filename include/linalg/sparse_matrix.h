#pragma once
#include <stdint.h>

#include "array.h"
#include "matrix.h"

/* Compressed Sparse Row pattern */
struct CSRPattern {
	TArray<uint32_t> row_start;
	TArray<uint32_t> col;
};

/* Compressed Sparse Row matrix */
struct CSRMatrix : public Matrix {
	bool symmetric = false;
	size_t nnz; /* Number of (non zero) entries */
	// Non zero entries on line i (0 <= i < rows)
	// are stored at indices row_start(i) <= k < row_start(i + 1).
	// Column indices are read into col(k), entry values
	// are read into data(k).
	uint32_t *row_start;
	uint32_t *col;
	TArray<double> data; /* Size = nnz  */
	void mvp(const double *__restrict x, double *__restrict y) const;
	double sum() const;
	double &operator()(uint32_t i, uint32_t j);
};
