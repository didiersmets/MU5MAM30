#pragma once
#include <stdint.h>

#include "array.h"
#include "matrix.h"

/******************************************************************************
 *
 * Compressed Sparse Row matrix
 *
 *****************************************************************************/

struct CSRPattern {
	bool symmetric;
	size_t rows;
	size_t cols;
	size_t nnz;
	/* Non zero entries on line i (0 <= i < rows)
	 * are stored at indices row_start(i) <= k < row_start(i + 1).
	 * Corresponding column indices are read into col(k). */
	TArray<uint32_t> row_start; /* Size = nrows + 1 */
	TArray<uint32_t> col; /* Size = nnz */
};

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

/* Visualisation of non zeros */
void spy(const CSRPattern &P, uint32_t width, const char *fname);

/******************************************************************************
 *
 * SKyLine sparse matrix
 *
 *****************************************************************************/

struct SKLPattern {
	TArray<size_t> row_start; /* Size = nrows + 1 */
	TArray<uint32_t> jmin; /* Size = nrows */
};

struct SKLMatrix : public Matrix {
	bool symmetric;
	size_t nnz; /* Number of (non zero) entries */
	/* Non zero entries on line i (0 <= i < rows) are stored at indices
	 * row_start(i) <= k < row_start(i + 1).
	 * Column idx j for nz coeffs on line i are those jmin(i) <= j <= i
	 */
	size_t *row_start;
	uint32_t *jmin;
	TArray<double> data; /* Size = nnz */
	void mvp(const double *__restrict x, double *__restrict y) const;
	double sum() const;
	double &operator()(uint32_t i, uint32_t j);
	void fwd_substitution(double *__restrict x,
			      const double *__restrict b) const;
	void bwd_substitution(double *__restrict x,
			      const double *__restrict b) const;
};

/* Visualisation of non zeros */
void spy(const SKLMatrix &L, uint32_t width, const char *fname);

