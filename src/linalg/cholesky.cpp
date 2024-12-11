#include <assert.h>
#include <math.h>
#include <algorithm>

#include "sparse_matrix.h"
#include "elimination_tree.h"
#include "math_utils.h"

/*******************************************************************************
 * CSR version
 ******************************************************************************/

void csr_build_elimination_tree(const CSRPattern &P, TArray<uint32_t> &Etree)
{
	/* Etree is just a parent relation */
	TArray<uint32_t> &parent = Etree;

	size_t n = P.rows;
	parent.resize(n);
	for (size_t i = 0; i < n; ++i) {
		Etree[i] = ~0u;
	}

	TArray<uint32_t> ancestor(n, ~0u);

	for (size_t i = 0; i < n; ++i) {
		size_t start = P.row_start[i];
		size_t stop = P.row_start[i + 1] - 1;
		for (size_t k = start; k < stop; ++k) {
			uint32_t j = P.col[k];
			uint32_t jroot = j;
			while (ancestor[jroot] != ~0u && ancestor[jroot] != i) {
				uint32_t tmp = ancestor[jroot];
				ancestor[jroot] = i;
				jroot = tmp;
			}
			if (ancestor[jroot] == ~0u) {
				ancestor[jroot] = parent[jroot] = i;
			}
		}
	}
}

void csr_build_cholesky_pattern(const CSRPattern &PA, CSRPattern &PL)
{
	assert(PA.symmetric);
	size_t n = PA.rows;

	PL.symmetric = true;
	PL.rows = PL.cols = n;
	PL.row_start.resize(n + 1);

	TArray<uint32_t> Etree;
	csr_build_elimination_tree(PA, Etree);

	TArray<uint32_t> mark(n);

	/* Counting pass */
	PL.row_start[0] = 0;
	for (size_t i = 0; i < n; ++i) {
		PL.row_start[i + 1] = 1; /* add 1 for diagonal */
		mark[i] = i;
		size_t start = PA.row_start[i];
		size_t stop = PA.row_start[i + 1] - 1;
		for (size_t k = start; k < stop; ++k) {
			uint32_t j = PA.col[k];
			while (mark[j] != i) {
				mark[j] = i;
				PL.row_start[i + 1]++;
				j = Etree[j];
			}
		}
	}

	/* Accumulate counts to finalize row_starts */
	for (size_t i = 0; i < n; ++i) {
		PL.row_start[i + 1] += PL.row_start[i];
	}

	/* Allocate and zero init output */
	PL.nnz = PL.row_start[n];
	PL.col.resize(PL.nnz);

	/* Fill pass 
	 * Note : 
	 *   - Fill-in cols j in line i may not be discovered in order 
	 *   - Therefore a posteriori sort
	 */
	TArray<uint32_t> offset(n);
	for (size_t i = 0; i < n; ++i) {
		/* Discover and record cols in line i */
		offset[i] = PL.row_start[i];
		mark[i] = i;
		size_t startA = PA.row_start[i];
		size_t stopA = PA.row_start[i + 1] - 1;
		for (size_t k = startA; k < stopA; ++k) {
			uint32_t j = PA.col[k];
			while (mark[j] != i) {
				mark[j] = i;
				PL.col[offset[i]++] = j;
				j = Etree[j];
			}
		}
		PL.col[offset[i]++] = i;
		assert(offset[i] == PL.row_start[i + 1]);
		/* Sort cols in line i */
		size_t startL = PL.row_start[i];
		size_t stopL = PL.row_start[i + 1] - 1;
		/* TODO rm std::sort with own heap sort */
		std::sort(&PL.col[startL], &PL.col[stopL + 1]);
	}
}

/* Sparse up-looking Cholesky factorization. 
 * Cfr. Scott & Tuma Algo 5.7 page 79
 */
void csr_cholesky_factorization(CSRMatrix &A, const CSRPattern &PL,
				CSRMatrix &L)
{
	uint32_t n = A.rows;
	A.data[0] = sqrt(A.data[0]);
	for (uint32_t i = 1; i < n; ++i) {
		/* Solve L_{upper block < i} */
	}
	/* TODO (finish) */
	(void)PL;
	(void)L;
	exit(0);
}

/*******************************************************************************
 * SkyLine version
 ******************************************************************************/

void in_place_cholesky_factorization(SKLMatrix &A)
{
	for (uint32_t i = 0; i < A.rows; ++i) {
		uint32_t jmini = A.jmin[i];
		double *Li = &A.data[A.row_start[i]] - jmini;
		for (uint32_t j = jmini; j < i; ++j) {
			uint32_t jminj = A.jmin[j];
			const double *Lj = &A.data[A.row_start[j]] - jminj;
			uint32_t jstart = MAX(jmini, jminj);
			double sum = 0.0;
			for (uint32_t k = jstart; k < j; ++k) {
				sum += Li[k] * Lj[k];
			}
			Li[j] = (Li[j] - sum) / Lj[j];
		}
		double sum = 0.0;
		for (uint32_t j = jmini; j < i; ++j) {
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
