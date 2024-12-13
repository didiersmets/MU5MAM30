#include <stdio.h>
#include <string.h>

#ifdef USE_OPENMP
	#include <omp.h>
#endif

#include "math_utils.h"
#include "P1.h"
#include "fem_matrix.h"
#include "adjacency.h"
#include "my_mesh.h"
#include "sparse_matrix.h"
#include "skyline.h"

void mvp_P1(const FEMatrix &A, const double *x, double *y)
{
	size_t vtx_count = A.m->vtx_count;
	size_t tri_count = A.m->tri_count;

#ifdef USE_OPENMP
	#pragma omp parallel for
#endif
	for (size_t v = 0; v < vtx_count; ++v) {
		y[v] = A.diag[v] * x[v];
	}

	// #pragma omp parallel for
	for (size_t t = 0; t < tri_count; ++t) {
		uint32_t a = A.m->triangles[t].x;
		uint32_t b = A.m->triangles[t].y;
		uint32_t c = A.m->triangles[t].z;
		double mult = A.off_diag[t];
		y[a] += mult * x[b];
		y[b] += mult * x[a];
		y[b] += mult * x[c];
		y[c] += mult * x[b];
		y[c] += mult * x[a];
		y[a] += mult * x[c];
	}
}

double sum_P1(const FEMatrix &A)
{
	size_t vtx_count = A.m->vtx_count;
	size_t tri_count = A.m->tri_count;

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
	size_t vtx_count = m.vtx_count;
	size_t tri_count = m.tri_count;

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

void inline mass(const Vec3d &AB, const Vec3d &AC, double *__restrict M)
{
	M[0] = norm(cross(AB, AC)) / 12;
	M[1] = M[0] / 2;
}

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


void build_P1_mass_matrix(const Mesh &m, FEMatrix &M)
{
	size_t vtx_count = m.vtx_count;
	size_t tri_count = m.tri_count;

	//M.fem_type = FEMatrix::P1_cst;
	M.m = &m;
	M.rows = M.cols = vtx_count;

	M.diag.resize(vtx_count);
	memset(M.diag.data, 0, vtx_count * sizeof(double));

	M.off_diag.resize(tri_count);
	
	for (size_t t = 0; t < tri_count; ++t) {
		uint32_t a = m.triangles[t].x;
		uint32_t b = m.triangles[t].y;
		uint32_t c = m.triangles[t].z;

		Vec3d A = m.vertices[a];
		Vec3d B = m.vertices[b];
		Vec3d C = m.vertices[c];
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
	size_t vtx_count = m.vtx_count;
	size_t tri_count = m.tri_count;
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

	
	for (size_t t = 0; t < tri_count; ++t) {
		uint32_t a = m.triangles[t].x;
		uint32_t b = m.triangles[t].y;
		uint32_t c = m.triangles[t].z;

		Vec3d A = m.vertices[a];
		Vec3d B = m.vertices[b];
		Vec3d C = m.vertices[c];
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
	size_t vtx_count = m.vtx_count;
	size_t tri_count = m.tri_count;

	//S.fem_type = FEMatrix::P1_sym;
	S.m = &m;
	S.rows = S.cols = vtx_count;

	S.diag.resize(vtx_count);
	memset(S.diag.data, 0, vtx_count * sizeof(double));

	S.off_diag.resize(3 * tri_count);
	
	for (size_t t = 0; t < tri_count; ++t) {
		uint32_t a = m.triangles[t].x;
		uint32_t b = m.triangles[t].y;
		uint32_t c = m.triangles[t].z;

		Vec3d A = m.vertices[a];
		Vec3d B = m.vertices[b];
		Vec3d C = m.vertices[c];
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
	size_t vtx_count = m.vtx_count;
	size_t tri_count = m.tri_count;
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

	
	for (size_t t = 0; t < tri_count; ++t) {
		uint32_t a = m.triangles[t].x;
		uint32_t b = m.triangles[t].y;
		uint32_t c = m.triangles[t].z;

		Vec3d A = m.vertices[a];
		Vec3d B = m.vertices[b];
		Vec3d C = m.vertices[c];
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

void build_P1_stiffness_matrix(const Mesh &m, SkylineMatrix &S){
	size_t N = m.vtx_count;
	size_t tri_count = m.tri_count;

	S.symmetric = true;
	S.rows = S.cols = N;
	
	S.J.resize(N);
	S.val.resize(9 * tri_count);
	S.start.resize(N);
	for (int i = 0; i < S.rows; i++) {
		S.J[i] = i;
	}
	
	//First I loop for all the triangles to fill the J array
	
	for (int i = 0; i < tri_count; i++){
		int a = m.triangles[i].x;
		int b = m.triangles[i].y;
		int c = m.triangles[i].z;
		assert(a < N);
		assert(b < N);
		assert(c < N);

		if(MIN(a,b) < S.J[MAX(a,b)]){
			S.J[MAX(a,b)] = MIN(a,b);
		}
		if(MIN(a,c) < S.J[MAX(a,c)]){
			S.J[MAX(a,c)] = MIN(a,c);
		}
		if(MIN(b,c) < S.J[MAX(b,c)]){
			S.J[MAX(b,c)] = MIN(b,c);
		}
	
	}
	
	//Then I fill the start array

	S.start[0] = 0;
	
	for (int i = 1; i < N; i++){
		S.start[i] = S.start[i-1] + i - S.J[i];
	}
	S.nnz = S.start[N - 1] + N - S.J[N - 1];
	S.val.resize(S.nnz);
	for (int i = 0; i < S.nnz; i++){
		S.val[i] = 0;
	}
	/* We fill the matrix S*/

	for (int i = 0; i < tri_count; i++) {
		int a = m.triangles[i].x;
		int b = m.triangles[i].y;
		int c = m.triangles[i].z;
		assert(a < N);
		assert(b < N);
		assert(c < N);
		Vec3d A = m.vertices[a];
		Vec3d B = m.vertices[b];
		Vec3d C = m.vertices[c];
		Vec3d AB = B - A;
		Vec3d BC = C - B;
		Vec3d CA = A - C;

		Vec3d CAxAB = cross(CA, AB);
		double area = 0.5 * norm(CAxAB);
		double r = 1. / (4 * area);
		S.val[S.start[a] + a - S.J[a]] += dot(BC, BC) * r;
		S.val[S.start[b] + b - S.J[b]] += dot(CA, CA) * r;
		S.val[S.start[c] + c - S.J[c]] += dot(AB, AB) * r;
		S.val[S.start[MIN(a,b)] + MAX(a,b) - S.J[MIN(a,b)]] += dot(CA, BC) * r;
		S.val[S.start[MIN(a,c)] + MAX(a,c) - S.J[MIN(a,c)]] += dot(AB, BC) * r;
		S.val[S.start[MIN(b,c)] + MAX(b,c) - S.J[MIN(b,c)]] += dot(AB, CA) * r;
	}
}
