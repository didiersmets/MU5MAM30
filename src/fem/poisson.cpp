#include "poisson.h"

#include "P1.h"
#include "array.h"
#include "conjugate_gradient.h"
#include "fem_matrix.h"
#include "mesh.h"
#include "tiny_blas.h"

PoissonSolver::PoissonSolver(const Mesh &m)
	: m(m)
	, N(m.vertex_count())
	, f(N)
	, u(N, 0.0)
	, r(N)
	, p(N)
	, Ap(N)
{
	build_P1_mass_matrix(m, M);
	build_P1_stiffness_matrix(m, A);
	vol = M.sum();
	inited = false;
	iterate = 0;
	converged = false;
}

void PoissonSolver::clear_solution()
{
	for (size_t i = 0; i < N; i++) {
		u[i] = 0;
	}
	init_cg();
	iterate = 0;
	converged = false;
}

void PoissonSolver::set_zero_mean(double *V)
{
	M.mvp(V, Ap.data);
	double s = blas_sum_in_place(Ap.data, N);
	for (size_t i = 0; i < N; ++i) {
		V[i] -= s / vol;
	}
}

void PoissonSolver::init_cg()
{
	double *F = f.data;
	double *U = u.data;
	double *R = r.data;
	double *P = p.data;
	double *AP = Ap.data;

	/* Fix up F and U for zero mean */
	set_zero_mean(F);
	set_zero_mean(U);

	/* r_0 = Mf - Au_0 */
	/* Compute also b2 = ||Mf||^2  */
	M.mvp(F, R);
	b2 = blas_dot(R, R, N);
	A.mvp(U, AP); /* AP used as temp storage */
	blas_axpy(-1, AP, R, N);
	/* p_0 = r_0 */
	blas_copy(R, P, N);
	r2 = blas_dot(R, R, N);
	rel_error = sqrt(r2 / b2);

	inited = true;
}

void PoissonSolver::do_iterate(size_t max_iter, double tol)
{
	if (!inited) {
		init_cg();
	}

	double *U = u.data;
	double *R = r.data;
	double *P = p.data;
	double *AP = Ap.data;

	while (max_iter-- && rel_error > tol) {
		r2 = cg_iterate_once(A, U, R, P, AP, r2);
		iterate++;
		rel_error = sqrt(r2 / b2);
	}

	if (rel_error <= tol)
		converged = true;
}
