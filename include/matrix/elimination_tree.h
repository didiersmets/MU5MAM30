#include "array.h"
#include "sparse_matrix.h"

void build_elimination_tree(const CSRMatrix &A, TArray<uint32_t> &parent);
size_t compute_fill_in(const CSRMatrix &A, const TArray<uint32_t> &parent);
