#pragma once

#include "vec3.h"

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
