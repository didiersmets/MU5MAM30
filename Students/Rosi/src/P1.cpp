#include <stdio.h>
#include <string.h>

#include "P1.h"
#include "fem_matrix.h"
#include "mesh.h"

void mvp_P1_cst(const FEMatrix &A, const double *x, double *y)
{
    assert(A.fem_type == FEMatrix::P1_cst);

    size_t vtx_count = A.m->vertex_count();
    size_t tri_count = A.m->triangle_count();
    const TArray<uint32_t> &idx = A.m->indices;

    for (size_t v = 0; v < vtx_count; ++v) {
        y[v] = A.diag[v] * x[v];
    }

    for (size_t t = 0; t < tri_count; ++t) {
        uint32_t a = idx[3 * t + 0];
        uint32_t b = idx[3 * t + 1];
        uint32_t c = idx[3 * t + 2];
        double mult = A.off_diag[t];
        y[a] += mult * x[b];
        y[b] += mult * x[a];
        y[b] += mult * x[c];
        y[c] += mult * x[b];
        y[c] += mult * x[a];
        y[a] += mult * x[c];
    }
}