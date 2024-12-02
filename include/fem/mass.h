#pragma once

#include "vec3.h"

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
