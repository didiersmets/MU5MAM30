#include <assert.h>

#include "array.h"
#include "sparse_matrix.h"

void build_elimination_tree(const CSRMatrix &A, TArray<uint32_t> &parent)
{
	assert(A.symmetric);
	size_t n = A.rows;

	parent.resize(n);
	for (size_t i = 0; i < n; ++i) {
		parent[i] = ~0u;
	}

	TArray<uint32_t> ancestor(n, ~0u);

	for (size_t i = 0; i < n; ++i) {
		size_t start = A.row_start[i];
		size_t stop = A.row_start[i + 1] - 1;
		for (size_t k = start; k < stop; ++k) {
			uint32_t j = A.col[k];
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

size_t compute_fill_in(const CSRMatrix &A, const TArray<uint32_t> &parent)
{
	size_t n = A.rows;
	TArray<uint32_t> mark(n);

	size_t fill_in = 0;
	for (size_t i = 0; i < n; ++i) {
		mark[i] = i;
		size_t start = A.row_start[i];
		size_t stop = A.row_start[i + 1] - 1;
		for (size_t k = start; k < stop; ++k) {
			uint32_t j = A.col[k];
			while (mark[j] != i) {
				mark[j] = i;
				fill_in++;
				j = parent[j];
			}
		}
	}
	return (fill_in + n);
}
