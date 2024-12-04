#include <string.h>

#include "P1.h"
#include "logging.h"
#include "cube.h"
#include "sparse_matrix.h"
#include "cholesky.h"
#include "mesh.h"
#include "sphere.h"
#include "chrono.h"

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

int main(int argc, char **argv)
{
	log_init(0);

	Mesh m;

	int res = -1;
	if (argc > 2 && strncmp(argv[1], "cube", 4) == 0) {
		res = load_cube(m, atoi(argv[2]));
	} else if (argc > 2 && strncmp(argv[1], "sphere", 5) == 0) {
		res = load_sphere(m, atoi(argv[2]));
	}
	if (res)
		exit(0);
	LOG_MSG("Loaded mesh.");
	LOG_MSG("Vertices : %zu. Triangles : %zu", m.vertex_count(),
		m.triangle_count());

	TArray<double> f(m.vertex_count());
	TArray<double> u(m.vertex_count());
	TArray<double> tmp(m.vertex_count());
	fill_rhs(m, f);

	SKLPattern P;
	SKLMatrix S;
	build_P1_SKLPattern(m, P);
	build_P1_stiffness_matrix(m, P, S);
	LOG_MSG("SKL stiffness matrix filled. NNz : %zu (%.1f * DOF^(3/2))",
		S.nnz, double(S.nnz) / pow(m.vertex_count(), 1.5));
	in_place_cholesky_decomposition(S);
	size_t rnnz = 0;
	for (size_t i = 0; i < S.nnz; ++i) {
		rnnz += (S.data[i] == 0 ? 0 : 1);
	}
	LOG_MSG("Actual NNZ : %zu", rnnz);
	LOG_MSG("Cholesky decomposition performed.");

	Timer chrono;
	chrono.start();
	S.fwd_substitution(tmp.data, f.data);
	chrono.stop("Fwd");
	chrono.start();
	S.bwd_substitution(u.data, tmp.data);
	chrono.stop("Bwd");
	chrono.start();
	cholesky_solve(S, f.data, u.data, tmp.data);
	chrono.stop("Cholesky solve");

	return (0);
}
