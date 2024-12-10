#include "fem_matrix.h"

//#include "P1.h"

static void mvp_P1_cst(const FEMatrix &A, const double *x, double *y);
static void mvp_P1_sym(const FEMatrix &A, const double *x, double *y);
static void mvp_P1_gen(const FEMatrix &A, const double *x, double *y);

void FEMatrix::mvp(const double *x, double *y) const
{
	switch (fem_type) {
	case FEMatrix::P1_cst:
		mvp_P1_cst(*this, x, y);
		return;
	case FEMatrix::P1_sym:
		mvp_P1_sym(*this, x, y);
		return;
	case FEMatrix::P1_gen:
		mvp_P1_gen(*this, x, y);
		return;
	}
}

static double sum_P1_cst(const FEMatrix &A);
static double sum_P1_sym(const FEMatrix &A);
static double sum_P1_gen(const FEMatrix &A);

double FEMatrix::sum() const
{
	switch (fem_type) {
	case FEMatrix::P1_cst:
		return sum_P1_cst(*this);
	case FEMatrix::P1_sym:
		return sum_P1_sym(*this);
	case FEMatrix::P1_gen:
		return sum_P1_gen(*this);
	default:
		return 0;
	}
}

static void mvp_P1_cst(const FEMatrix &A, const double *x, double *y)
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

void mvp_P1_sym(const FEMatrix &A, const double *x, double *y)
{
	assert(A.fem_type == FEMatrix::P1_sym);

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
		y[a] += A.off_diag[3 * t + 0] * x[b];
		y[b] += A.off_diag[3 * t + 0] * x[a];
		y[b] += A.off_diag[3 * t + 1] * x[c];
		y[c] += A.off_diag[3 * t + 1] * x[b];
		y[c] += A.off_diag[3 * t + 2] * x[a];
		y[a] += A.off_diag[3 * t + 2] * x[c];
	}
}

void mvp_P1_gen(const FEMatrix &A, const double *x, double *y)
{
	assert(A.fem_type == FEMatrix::P1_sym);

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
		y[a] += A.off_diag[6 * t + 0] * x[b];
		y[b] += A.off_diag[6 * t + 1] * x[a];
		y[b] += A.off_diag[6 * t + 2] * x[c];
		y[c] += A.off_diag[6 * t + 3] * x[b];
		y[c] += A.off_diag[6 * t + 4] * x[a];
		y[a] += A.off_diag[6 * t + 5] * x[c];
	}
}

double sum_P1_cst(const FEMatrix &A)
{
	assert(A.fem_type == FEMatrix::P1_cst);

	size_t vtx_count = A.m->vertex_count();
	size_t tri_count = A.m->triangle_count();

	double sum1 = 0.0;
	for (size_t v = 0; v < vtx_count; ++v) {
		sum1 += A.diag[v];
	}

	double sum2 = 0.0;
	for (size_t t = 0; t < tri_count; ++t) {
		sum2 += 6 * A.off_diag[t];
	}

	return sum1 + sum2;
}

double sum_P1_sym(const FEMatrix &A)
{
	assert(A.fem_type == FEMatrix::P1_sym);

	size_t vtx_count = A.m->vertex_count();
	size_t tri_count = A.m->triangle_count();

	double sum1 = 0.0;
	for (size_t v = 0; v < vtx_count; ++v) {
		sum1 += A.diag[v];
	}

	double sum2 = 0;
	for (size_t t = 0; t < tri_count; ++t) {
		sum2 += 2 * A.off_diag[3 * t + 0];
		sum2 += 2 * A.off_diag[3 * t + 1];
		sum2 += 2 * A.off_diag[3 * t + 3];
	}

	return sum1 + sum2;
}

double sum_P1_gen(const FEMatrix &A)
{
	assert(A.fem_type == FEMatrix::P1_sym);

	size_t vtx_count = A.m->vertex_count();
	size_t tri_count = A.m->triangle_count();

	double sum1 = 0.0;
	for (size_t v = 0; v < vtx_count; ++v) {
		sum1 += A.diag[v];
	}

	double sum2 = 0.0;
	for (size_t t = 0; t < tri_count; ++t) {
		sum2 += A.off_diag[6 * t + 0];
		sum2 += A.off_diag[6 * t + 1];
		sum2 += A.off_diag[6 * t + 2];
		sum2 += A.off_diag[6 * t + 3];
		sum2 += A.off_diag[6 * t + 4];
		sum2 += A.off_diag[6 * t + 5];
	}

	return sum1 + sum2;
}
