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
