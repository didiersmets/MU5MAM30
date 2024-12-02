#pragma once
#include <stdint.h>

#include "array.h"
#include "matrix.h"

/* Compressed Sparse Row pattern */
struct CSRPattern {
	// Non zero entries on line i (0 <= i < rows)
	// are stored at indices row_start(i) <= k < row_start(i + 1).
	// Corresponding column indices are read into col(k).
	TArray<uint32_t> row_start; /* Size = nrows + 1 */
	TArray<uint32_t> col; /* Size = nnz */
};

/* Compressed Sparse Row matrix */
struct CSRMatrix : public Matrix {
	bool symmetric = false;
	size_t nnz; /* Number of (non zero) entries */
	uint32_t *row_start; /* pointer to the corresponding data in pattern */
	uint32_t *col; /* pointer to the corresponding data in pattern */
	TArray<double> data; /* Size = nnz  */
	void mvp(const double *__restrict x, double *__restrict y) const;
	double sum() const;
	double &operator()(uint32_t i, uint32_t j);
};

/* Skyline Lower triangular pattern */
struct SKLPattern {
	TArray<size_t> row_start; /* Size = nrows + 1 */
	TArray<uint32_t> jmin; /* Size = nrows */
};

/* Skyline Lower triangular matrix */
struct SKLMatrix : public Matrix {
	size_t nnz; /* Number of (non zero) entries */
	// Non zero entries on line i (0 <= i < rows) are stored at indices
	// row_start(i) <= k < row_start(i + 1).
	// Column indices j for nnz coeffs on line i are those jmin(i) <= j <= i
	size_t *row_start;
	uint32_t *jmin;
	TArray<double> data; /* Size = nnz */
	double &operator()(uint32_t i, uint32_t j);
	void fwd_substitution(double *__restrict x,
			      const double *__restrict b) const;
	void bwd_substitution(double *__restrict x,
			      const double *__restrict b) const;
};

