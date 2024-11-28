#include <stdio.h>
#include <string.h>

#ifdef USE_OPENMP
	#include <omp.h>
#endif

#include "P1.h"
#include "adjacency.h"
#include "fem_matrix.h"
#include "mesh.h"
#include "sparse_matrix.h"

void mvp_P1_cst(const FEMatrix &A, const double *x, double *y)
{
	assert(A.fem_type == FEMatrix::P1_cst);

	size_t vtx_count = A.m->vertex_count();
	size_t tri_count = A.m->triangle_count();
	const TArray<uint32_t> &idx = A.m->indices;

#ifdef USE_OPENMP
	#pragma omp parallel for
#endif
	for (size_t v = 0; v < vtx_count; ++v) {
		y[v] = A.diag[v] * x[v];
	}

	// #pragma omp parallel for
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

double sum_P1_cst(const FEMatrix &A)
{
	assert(A.fem_type == FEMatrix::P1_cst);

	size_t vtx_count = A.m->vertex_count();
	size_t tri_count = A.m->triangle_count();

	double sum1 = 0.0;
#ifdef USE_OPENMP
	#pragma omp parallel for reduction(+ : sum1)
#endif
	for (size_t v = 0; v < vtx_count; ++v) {
		sum1 += A.diag[v];
	}

	double sum2 = 0.0;
#ifdef USE_OPENMP
	#pragma omp parallel for reduction(+ : sum2)
#endif
	for (size_t t = 0; t < tri_count; ++t) {
		sum2 += 6 * A.off_diag[t];
	}

	return sum1 + sum2;
}

void mvp_P1_sym(const FEMatrix &A, const double *x, double *y)
{
	assert(A.fem_type == FEMatrix::P1_sym);

	size_t vtx_count = A.m->vertex_count();
	size_t tri_count = A.m->triangle_count();
	const TArray<uint32_t> &idx = A.m->indices;

#ifdef USE_OPENMP
	#pragma omp parallel for
#endif
	for (size_t v = 0; v < vtx_count; ++v) {
		y[v] = A.diag[v] * x[v];
	}

	// #pragma omp parallel for
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

double sum_P1_sym(const FEMatrix &A)
{
	assert(A.fem_type == FEMatrix::P1_sym);

	size_t vtx_count = A.m->vertex_count();
	size_t tri_count = A.m->triangle_count();

	double sum1 = 0.0;
#ifdef USE_OPENMP
	#pragma omp parallel for reduction(+ : sum1)
#endif
	for (size_t v = 0; v < vtx_count; ++v) {
		sum1 += A.diag[v];
	}

	double sum2 = 0;
#ifdef USE_OPENMP
	#pragma omp parallel for reduction(+ : sum2)
#endif
	for (size_t t = 0; t < tri_count; ++t) {
		sum2 += 2 * A.off_diag[3 * t + 0];
		sum2 += 2 * A.off_diag[3 * t + 1];
		sum2 += 2 * A.off_diag[3 * t + 3];
	}

	return sum1 + sum2;
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

double sum_P1_gen(const FEMatrix &A)
{
	assert(A.fem_type == FEMatrix::P1_sym);

	size_t vtx_count = A.m->vertex_count();
	size_t tri_count = A.m->triangle_count();

	double sum1 = 0.0;
#ifdef USE_OPENMP
	#pragma omp parallel for reduction(+ : sum1)
#endif
	for (size_t v = 0; v < vtx_count; ++v) {
		sum1 += A.diag[v];
	}

	double sum2 = 0.0;
#ifdef USE_OPENMP
	#pragma omp parallel for reduction(+ : sum2)
#endif
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
		uint32_t *to_sort = &P.col[P.row_start[a]];
		size_t count = P.row_start[a + 1] - P.row_start[a];
		/* Insertion sort */
		for (size_t k = 1; k < count; ++k) {
			size_t j = k - 1;
			while (j && (to_sort[j] > to_sort[j + 1])) {
				uint32_t tmp = to_sort[j];
				to_sort[j] = to_sort[j + 1];
				to_sort[j + 1] = tmp;
				j--;
			}
		}
	}
}

static void stiffness(const Vec3d &AB, const Vec3d &AC, double *__restrict S);
static void mass(const Vec3d &AB, const Vec3d &AC, double *__restrict M);

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
		Vec3d AB = {(double)B[0] - (double)A[0],
			    (double)B[1] - (double)A[1],
			    (double)B[2] - (double)A[2]};
		Vec3d AC = {(double)C[0] - (double)A[0],
			    (double)C[1] - (double)A[1],
			    (double)C[2] - (double)A[2]};
		double Mloc[2];
		mass(AB, AC, Mloc);
		M.diag[a] += Mloc[0];
		M.diag[b] += Mloc[0];
		M.diag[c] += Mloc[0];
		M.off_diag[t] = Mloc[1];
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
		Vec3d AB = {(double)B[0] - (double)A[0],
			    (double)B[1] - (double)A[1],
			    (double)B[2] - (double)A[2]};
		Vec3d AC = {(double)C[0] - (double)A[0],
			    (double)C[1] - (double)A[1],
			    (double)C[2] - (double)A[2]};
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
		Vec3d AB = {(double)B[0] - (double)A[0],
			    (double)B[1] - (double)A[1],
			    (double)B[2] - (double)A[2]};
		Vec3d AC = {(double)C[0] - (double)A[0],
			    (double)C[1] - (double)A[1],
			    (double)C[2] - (double)A[2]};
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
		Vec3d AB = {(double)B[0] - (double)A[0],
			    (double)B[1] - (double)A[1],
			    (double)B[2] - (double)A[2]};
		Vec3d AC = {(double)C[0] - (double)A[0],
			    (double)C[1] - (double)A[1],
			    (double)C[2] - (double)A[2]};
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

/* Given a triangle ABC, computes the (symmetric) 3x3 mass M s.t.
 *
 *   M_{ij} := \int_{ABC} \phi_i \phi_j
 *
 * where \phi_0 := \phi_A, \phi_1 := \phi_B, \phi_2 := \phi_C
 * are the shape functions of the P1 Lagrange element associated
 * to ABC.
 *
 * Idea behind computation :
 * -------------------------
 *
 * We denote by \Psi the affine map
 *
 *    \Psi(s,t) = sB + tC + (1-s-t)A.
 *
 * Then \Psi maps the reference simplex in R^2 (we denote it by A'B'C')
 * onto ABC, and since \Psi is affine \phi_X = Psi \circ \phi_X' for
 * any X in {A, B, C}. Moreover by the change of variable formula, for
 * arbitrary X, Y in {A, B, C} :
 *
 *    \int_{ABC} \phi_X \phi_Y = \int_{A'B'C'} \phi_X' \phi_Y' |Jac(\Psi)|dsdt
 *
 * where the Jacobian |Jac(\Psi)| is constant equal to |ABC|/|A'B'C'| = 2|ABC|.
 *
 * Besides, elementary integration shows that
 *
 *               (2  1  1)
 * M' = (1/24) * (1  2  1)
 *               (1  1  2)
 *
 * We therefore only return |ABC|/6 and |ABC|/12, with |ABC| = |AB x AC| / 2.
 */
void inline mass(const Vec3d &AB, const Vec3d &AC, double *__restrict M)
{
	M[0] = norm(cross(AB, AC)) / 12;
	M[1] = M[0] / 2;
}

/* Given a triangle ABC, computes the (symmetric) 3x3 stiffness matrix S s.t.
 *
 *   S_{ij} := \int_{ABC} \nabla \phi_i \cdot \nabla \phi_j
 *
 * where \phi_0 := \phi_A, \phi_1 := \phi_B, \phi_2 := \phi_C
 * are the shape functions of the P1 Lagrange element associated
 * to ABC.
 *
 * Input : the vectors AB and AC.
 * Output: the six coefficients S_{00} S_{11} S_{22} S_{01} S_{12} S_{20},
 *         corresponding to the interactions A<->A, B<->B, C<->C, A<->B, B<->C,
 *         C<->A
 *
 * Idea behind computation :
 * -------------------------
 *
 * We denote by a, b, c the angles at A, B, C; and by n_A, n_B, n_C the
 * inward normals to the segments opposite to A, B, C.
 *
 * We have :
 *
 *     \nabla \phi_B = 1/(|AB|sin(a)) * n_B
 *     \nabla \phi_C = 1/(|CA|sin(a)) * n_C
 *     n_B \cdot n_C = -cos(a)
 *     2|ABC| = |CA x AB| = |CA| * |AB| * sin(a)
 *     dot(CA, AB) = -|CA| |AB| cos(a)
 *
 * hence :
 *
 *     S_{BC} = -1/2 * cot(a) =  dot(CA, AB) / (4|ABC|)
 *
 * and similarly for S_{AB} and S_{CA}.
 *
 * Also :
 *
 *     \nabla \phi_A = 1/(|AB|sin(b)) * n_A
 *                   = 1/(|CA|sin(c)) * n_A
 *
 * hence :
 *
 *     S_{A,A} = |ABC| / (|AB||CA|sin(b)sin(c)) = |BC|^2 / (4|ABC|)
 *
 * and similarly for S_{BB} and S_{CC}.
 *
 * Taking into account that BC = AC - AB, we simplify the above expressions
 * into the following.
 */
void inline stiffness(const Vec3d &AB, const Vec3d &AC, double *__restrict S)
{
	double ABAB = norm2(AB);
	double ACAC = norm2(AC);
	double ABAC = dot(AB, AC);
	double mult = 0.5 / sqrt(ABAB * ACAC - ABAC * ABAC);
	ABAB *= mult;
	ACAC *= mult;
	ABAC *= mult;

	S[0] = ACAC - 2 * ABAC + ABAB;
	S[1] = ACAC;
	S[2] = ABAB;
	S[3] = ABAC - ACAC;
	/* Note the chosen order : (B,C)-> 4 and (C,A) -> 5 */
	S[4] = -ABAC;
	S[5] = ABAC - ABAB;
}
