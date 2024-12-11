#include "array.h"
#include "sparse_matrix.h"

void csr_build_elimination_tree(const CSRPattern &P, TArray<uint32_t> &Etree);
void csr_build_cholesky_pattern(const CSRPattern &PA, CSRPattern &PL);
void csr_cholesky_factorization(const CSRMatrix &A, const CSRPattern &PL,
				CSRMatrix &L);

void in_place_cholesky_factorization(SKLMatrix &A);
void cholesky_solve(const SKLMatrix &L, const double *__restrict b,
		    double *__restrict x, double *__restrict tmp);
