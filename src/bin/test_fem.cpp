#include <string.h>

#include "P1.h"
#include "chrono.h"
#include "conjugate_gradient.h"
#include "cube.h"
#include "fem_matrix.h"
#include "mesh.h"
#include "mesh_io.h"
#include "sphere.h"
#include "tiny_blas.h"

void syntax(char *prg_name)
{
	printf("Syntax : %s ($(obj_filename)| cube | sphere) [n]\n", prg_name);
	printf("         Subdivision number n must be provided in case of\n"
	       "         cube or sphere mesh.\n");
	exit(EXIT_FAILURE);
}

static double test_f(Vec3 pos)
{
	float x = pos[0];
	float y = pos[1];
	return (5 * pow(x, 4) * y - 10 * pow(x, 2) * pow(y, 3) + pow(y, 5));
}

static void fill_rhs(const Mesh &mesh, TArray<double> &f)
{
	for (size_t i = 0; i < mesh.vertex_count(); ++i) {
		f[i] = test_f(mesh.positions[i]);
	}
}

static void add_mass_to_stiffness(FEMatrix &S, const FEMatrix &M)
{
	const Mesh *m = S.m;

	for (size_t i = 0; i < m->vertex_count(); ++i) {
		S.diag[i] += M.diag[i];
	}
	for (size_t i = 0; i < m->triangle_count(); ++i) {
		S.off_diag[3 * i + 0] += M.off_diag[i];
		S.off_diag[3 * i + 1] += M.off_diag[i];
		S.off_diag[3 * i + 2] += M.off_diag[i];
	}
}

/* Solves -\Delta u + u = f */
static size_t system_solve(const Mesh &m, const TArray<double> &f,
			   TArray<double> &u, int max_iter)
{
	size_t N = m.vertex_count();
	assert(u.size == N && f.size == N);

	TArray<double> b(N);
	TArray<double> r(N);
	TArray<double> p(N);
	TArray<double> Ap(N);

	FEMatrix S;
	FEMatrix M;

	build_P1_mass_matrix(m, M);
	build_P1_stiffness_matrix(m, S);
	add_mass_to_stiffness(S, M);

	M.mvp(f.data, b.data);

	double relative_error;
	size_t iter = conjugate_gradient_solve(S, b.data, u.data, r.data,
					       p.data, Ap.data, &relative_error,
					       1e-6, max_iter);

	return (iter);
}

int main(int argc, char **argv)
{
	Timer chrono;
	Mesh mesh;

	chrono.start();
	int res = -1;
	if (argc > 2 && strncmp(argv[1], "cube", 4) == 0) {
		res = load_cube(mesh, atoi(argv[2]));
	} else if (argc > 2 && strncmp(argv[1], "sphere", 5) == 0) {
		res = load_sphere(mesh, atoi(argv[2]));
	} else if (argc > 1) {
		res = load_obj(argv[1], mesh);
	}
	if (res)
		syntax(argv[0]);
	printf("Vertices : %zu. Triangles : %zu\n", mesh.vertex_count(),
	       mesh.triangle_count());
	chrono.stop("loading mesh");

	chrono.start();
	TArray<double> f(mesh.vertex_count());
	TArray<double> u(mesh.vertex_count(), 0.0);

	fill_rhs(mesh, f);
	system_solve(mesh, f, u, -1);
	chrono.stop("Laplace solve");

	return (0);
}

