#include <stdio.h>
#include <string.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "P1.h"
#include "adjacency.h"
#include "fem_matrix.h"
#include "mass.h"
#include "mesh.h"
#include "sparse_matrix.h"
#include "stiffness.h"

/* FEMatrix variants */

void build_P1_mass_matrix(const Mesh &m, FEMatrix &M)
{
	size_t vtx_count = m.vertex_count();
	size_t tri_count = m.triangle_count();

	M.fem_type = FEMatrix::P1_cst;
	M.m = &m;
	M.rows = M.cols = vtx_count;

	M.diag.resize(vtx_count);
	memset(M.diag.data, 0, vtx_count * sizeof(double));

	M.off_diag.resize(tri_count);
	const TArray<uint32_t> &idx = m.indices;
	for (size_t t = 0; t < tri_count; ++t) {
		uint32_t a = idx[3 * t + 0];
		uint32_t b = idx[3 * t + 1];
		uint32_t c = idx[3 * t + 2];
		Vec3f A = m.positions[a];
		Vec3f B = m.positions[b];
		Vec3f C = m.positions[c];
		Vec3d AB = { (double)B[0] - (double)A[0],
			     (double)B[1] - (double)A[1],
			     (double)B[2] - (double)A[2] };
		Vec3d AC = { (double)C[0] - (double)A[0],
			     (double)C[1] - (double)A[1],
			     (double)C[2] - (double)A[2] };
		double Mloc[2];
		mass(AB, AC, Mloc);
		M.diag[a] += Mloc[0];
		M.diag[b] += Mloc[0];
		M.diag[c] += Mloc[0];
		M.off_diag[t] = Mloc[1];
	}
}

void build_P1_stiffness_matrix(const Mesh &m, FEMatrix &S)
{
	size_t vtx_count = m.vertex_count();
	size_t tri_count = m.triangle_count();

	S.fem_type = FEMatrix::P1_sym;
	S.m = &m;
	S.rows = S.cols = vtx_count;

	S.diag.resize(vtx_count);
	memset(S.diag.data, 0, vtx_count * sizeof(double));

	S.off_diag.resize(3 * tri_count);
	const TArray<uint32_t> &idx = m.indices;
	for (size_t t = 0; t < tri_count; ++t) {
		uint32_t a = idx[3 * t + 0];
		uint32_t b = idx[3 * t + 1];
		uint32_t c = idx[3 * t + 2];
		Vec3f A = m.positions[a];
		Vec3f B = m.positions[b];
		Vec3f C = m.positions[c];
		Vec3d AB = { (double)B[0] - (double)A[0],
			     (double)B[1] - (double)A[1],
			     (double)B[2] - (double)A[2] };
		Vec3d AC = { (double)C[0] - (double)A[0],
			     (double)C[1] - (double)A[1],
			     (double)C[2] - (double)A[2] };
		double Sloc[6];
		stiffness(AB, AC, Sloc);
		S.diag[a] += Sloc[0];
		S.diag[b] += Sloc[1];
		S.diag[c] += Sloc[2];
		S.off_diag[3 * t + 0] = Sloc[3];
		S.off_diag[3 * t + 1] = Sloc[4];
		S.off_diag[3 * t + 2] = Sloc[5];
	}
}

/* CSRMatrix variants */

static bool find(uint32_t x, uint32_t *start, size_t count)
{
	for (size_t i = 0; i < count; ++i) {
		if (start[i] == x)
			return true;
	}
	return false;
}

void build_P1_CSRPattern(const Mesh &m, CSRPattern &P)
{
	size_t vtx_count = m.vertex_count();
	size_t tri_count = m.triangle_count();

	P.symmetric = true;
	P.rows = P.cols = vtx_count;
	P.row_start.resize(vtx_count + 1);

	VTAdjacency adj(m);

	/* Upper bound on the number of edges a->b with a <= b */
	size_t max_nnz = 3 * tri_count + vtx_count;
	P.col.resize(max_nnz);

	/* Fill P.row_start and P.col (not yet ordered) */
	size_t nnz = 0;
	for (size_t a = 0; a < vtx_count; ++a) {
		P.row_start[a] = nnz;
		uint32_t *start = &P.col[nnz];
		size_t nnz_loc = 0;
		uint32_t kstart = adj.offset[a];
		uint32_t kstop = kstart + adj.degree[a];
		for (size_t k = kstart; k < kstop; ++k) {
			uint32_t b = adj.vtri[k].next;
			uint32_t c = adj.vtri[k].prev;
			if (b < a && !find(b, start, nnz_loc)) {
				P.col[nnz++] = b;
				nnz_loc++;
			}
			if (c < a && !find(c, start, nnz_loc)) {
				P.col[nnz++] = c;
				nnz_loc++;
			}
		}
		P.col[nnz++] = a;
	}
	P.row_start[vtx_count] = nnz;
	P.col.resize(nnz);
	P.col.shrink_to_fit();

	/* Reorder each "line" of P.col in increasing column order */
	for (size_t a = 0; a < vtx_count; ++a) {
		uint32_t *__restrict to_sort = &P.col[P.row_start[a]];
		size_t count = P.row_start[a + 1] - P.row_start[a];
		/* Insertion sort (small degree vtx) */
		for (size_t k = 1; k < count; ++k) {
			size_t j = k;
			while (j && (to_sort[j - 1] > to_sort[j])) {
				uint32_t tmp = to_sort[j - 1];
				to_sort[j - 1] = to_sort[j];
				to_sort[j] = tmp;
				j--;
			}
		}
	}
}

void build_P1_mass_matrix(const Mesh &m, const CSRPattern &P, CSRMatrix &M)
{
	size_t vtx_count = m.vertex_count();
	size_t tri_count = m.triangle_count();
	assert(P.row_start.size == vtx_count + 1);

	M.symmetric = true;
	M.rows = M.cols = vtx_count;
	M.nnz = P.col.size;
	M.row_start = P.row_start.data;
	M.col = P.col.data;
	M.data.resize(M.nnz);
	for (size_t i = 0; i < M.nnz; ++i) {
		M.data[i] = 0.0;
	}

	const TArray<uint32_t> &idx = m.indices;
	for (size_t t = 0; t < tri_count; ++t) {
		uint32_t a = idx[3 * t + 0];
		uint32_t b = idx[3 * t + 1];
		uint32_t c = idx[3 * t + 2];
		Vec3f A = m.positions[a];
		Vec3f B = m.positions[b];
		Vec3f C = m.positions[c];
		Vec3d AB = { (double)B[0] - (double)A[0],
			     (double)B[1] - (double)A[1],
			     (double)B[2] - (double)A[2] };
		Vec3d AC = { (double)C[0] - (double)A[0],
			     (double)C[1] - (double)A[1],
			     (double)C[2] - (double)A[2] };
		double Mloc[2];
		mass(AB, AC, Mloc);
		M(a, a) += Mloc[0];
		M(b, b) += Mloc[0];
		M(c, c) += Mloc[0];
		M(a > b ? a : b, a > b ? b : a) += Mloc[1];
		M(b > c ? b : c, b > c ? c : b) += Mloc[1];
		M(c > a ? c : a, c > a ? a : c) += Mloc[1];
	}
}

void build_P1_stiffness_matrix(const Mesh &m, const CSRPattern &P, CSRMatrix &S)
{
	size_t vtx_count = m.vertex_count();
	size_t tri_count = m.triangle_count();
	assert(P.row_start.size == vtx_count + 1);

	S.symmetric = true;
	S.rows = S.cols = vtx_count;
	S.nnz = P.col.size;
	S.row_start = P.row_start.data;
	S.col = P.col.data;
	S.data.resize(S.nnz);
	for (size_t i = 0; i < S.nnz; ++i) {
		S.data[i] = 0.0;
	}

	const TArray<uint32_t> &idx = m.indices;
	for (size_t t = 0; t < tri_count; ++t) {
		uint32_t a = idx[3 * t + 0];
		uint32_t b = idx[3 * t + 1];
		uint32_t c = idx[3 * t + 2];
		Vec3f A = m.positions[a];
		Vec3f B = m.positions[b];
		Vec3f C = m.positions[c];
		Vec3d AB = { (double)B[0] - (double)A[0],
			     (double)B[1] - (double)A[1],
			     (double)B[2] - (double)A[2] };
		Vec3d AC = { (double)C[0] - (double)A[0],
			     (double)C[1] - (double)A[1],
			     (double)C[2] - (double)A[2] };
		double Sloc[6];
		stiffness(AB, AC, Sloc);
		S(a, a) += Sloc[0];
		S(b, b) += Sloc[1];
		S(c, c) += Sloc[2];
		S(a > b ? a : b, a > b ? b : a) += Sloc[3];
		S(b > c ? b : c, b > c ? c : b) += Sloc[4];
		S(c > a ? c : a, c > a ? a : c) += Sloc[5];
	}
}

/* SKLMatrix variants */

void build_P1_SKLPattern(const Mesh &m, SKLPattern &P)
{
	size_t vtx_count = m.vertex_count();
	size_t tri_count = m.triangle_count();

	P.jmin.resize(vtx_count);
	for (size_t v = 0; v < vtx_count; ++v) {
		P.jmin[v] = v;
	}
	for (size_t t = 0; t < tri_count; t++) {
		uint32_t a = m.indices[3 * t + 0];
		uint32_t b = m.indices[3 * t + 1];
		uint32_t c = m.indices[3 * t + 2];
		P.jmin[a] = b < P.jmin[a] ? b : P.jmin[a];
		P.jmin[a] = c < P.jmin[a] ? c : P.jmin[a];
		P.jmin[b] = a < P.jmin[b] ? a : P.jmin[b];
		P.jmin[b] = c < P.jmin[b] ? c : P.jmin[b];
		P.jmin[c] = b < P.jmin[c] ? b : P.jmin[c];
		P.jmin[c] = a < P.jmin[c] ? a : P.jmin[c];
	}
	P.row_start.resize(vtx_count + 1);
	uint32_t start = 0;
	for (size_t v = 0; v < vtx_count; v++) {
		P.row_start[v] = start;
		start += v - P.jmin[v] + 1;
	}
	P.row_start[vtx_count] = start;
}

void build_P1_mass_matrix(const Mesh &m, const SKLPattern &P, SKLMatrix &M)
{
	size_t vtx_count = m.vertex_count();
	size_t tri_count = m.triangle_count();
	assert(P.row_start.size == vtx_count + 1);

	M.symmetric = true;
	M.rows = M.cols = vtx_count;
	M.nnz = P.row_start[vtx_count];
	M.row_start = P.row_start.data;
	M.jmin = P.jmin.data;
	M.data.resize(M.nnz);
	for (size_t i = 0; i < M.nnz; ++i) {
		M.data[i] = 0.0;
	}

	const TArray<uint32_t> &idx = m.indices;
	for (size_t t = 0; t < tri_count; ++t) {
		uint32_t a = idx[3 * t + 0];
		uint32_t b = idx[3 * t + 1];
		uint32_t c = idx[3 * t + 2];
		Vec3f A = m.positions[a];
		Vec3f B = m.positions[b];
		Vec3f C = m.positions[c];
		Vec3d AB = { (double)B[0] - (double)A[0],
			     (double)B[1] - (double)A[1],
			     (double)B[2] - (double)A[2] };
		Vec3d AC = { (double)C[0] - (double)A[0],
			     (double)C[1] - (double)A[1],
			     (double)C[2] - (double)A[2] };
		double Mloc[2];
		mass(AB, AC, Mloc);
		M(a, a) += Mloc[0];
		M(b, b) += Mloc[0];
		M(c, c) += Mloc[0];
		M(a > b ? a : b, a > b ? b : a) += Mloc[1];
		M(b > c ? b : c, b > c ? c : b) += Mloc[1];
		M(c > a ? c : a, c > a ? a : c) += Mloc[1];
	}
}

void build_P1_stiffness_matrix(const Mesh &m, const SKLPattern &P, SKLMatrix &S)
{
	size_t vtx_count = m.vertex_count();
	size_t tri_count = m.triangle_count();

	S.symmetric = true;
	S.rows = S.cols = vtx_count;
	S.nnz = P.row_start[vtx_count];
	S.row_start = P.row_start.data;
	S.jmin = P.jmin.data;
	S.data.resize(S.nnz);
	for (size_t i = 0; i < S.nnz; ++i) {
		S.data[i] = 0.0;
	}

	const TArray<uint32_t> &idx = m.indices;
	for (size_t t = 0; t < tri_count; ++t) {
		uint32_t a = idx[3 * t + 0];
		uint32_t b = idx[3 * t + 1];
		uint32_t c = idx[3 * t + 2];
		Vec3f A = m.positions[a];
		Vec3f B = m.positions[b];
		Vec3f C = m.positions[c];
		Vec3d AB = { (double)B[0] - (double)A[0],
			     (double)B[1] - (double)A[1],
			     (double)B[2] - (double)A[2] };
		Vec3d AC = { (double)C[0] - (double)A[0],
			     (double)C[1] - (double)A[1],
			     (double)C[2] - (double)A[2] };
		double Sloc[6];
		stiffness(AB, AC, Sloc);
		S(a, a) += Sloc[0];
		S(b, b) += Sloc[1];
		S(c, c) += Sloc[2];
		S(a > b ? a : b, a > b ? b : a) += Sloc[3];
		S(b > c ? b : c, b > c ? c : b) += Sloc[4];
		S(c > a ? c : a, c > a ? a : c) += Sloc[5];
	}
}

