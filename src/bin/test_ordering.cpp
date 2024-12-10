#include <string.h>

#include "P1.h"
#include "logging.h"
#include "cube.h"
#include "sparse_matrix.h"
#include "cholesky.h"
#include "mesh.h"

int main(int argc, char **argv)
{
	log_init(0);

	Mesh m1, m2;

	if (argc > 1) {
		uint32_t subdiv = atoi(argv[1]);
		load_cube(m1, subdiv);
		LOG_MSG("Loaded standard cube.");
		LOG_MSG("Vertices : %zu. Triangles : %zu", m1.vertex_count(),
			m1.triangle_count());
		load_cube_nested_dissect(m2, subdiv);
		LOG_MSG("Loaded dissected cube.");
		LOG_MSG("Vertices : %zu. Triangles : %zu", m2.vertex_count(),
			m2.triangle_count());
	} else {
		return (EXIT_FAILURE);
	}

	SKLPattern P1;
	SKLMatrix S1;
	build_P1_SKLPattern(m1, P1);
	build_P1_stiffness_matrix(m1, P1, S1);
	LOG_MSG("SKL stiffness matrix filled. NNz1 : %zu (%.1f * DOF^(3/2))",
		S1.nnz, double(S1.nnz) / pow(m1.vertex_count(), 1.5));
	in_place_cholesky_decomposition(S1);
	size_t rnnz1 = 0;
	for (size_t i = 0; i < S1.nnz; ++i) {
		rnnz1 += (S1.data[i] == 0 ? 0 : 1);
	}
	LOG_MSG("Actual NNZ1 : %zu (%.1f\%, %.3f * DOF^(3/2))", rnnz1,
		(float)rnnz1 / S1.nnz * 100,
		(double)rnnz1 / pow(m1.vertex_count(), 1.5));
	spy(S1, 1024, "spy_normal_cube.png");

	SKLPattern P2;
	SKLMatrix S2;
	build_P1_SKLPattern(m2, P2);
	build_P1_stiffness_matrix(m2, P2, S2);
	LOG_MSG("SKL stiffness matrix filled. NNz2 : %zu (%.1f * DOF^(3/2))",
		S2.nnz, double(S2.nnz) / pow(m2.vertex_count(), 1.5));
	in_place_cholesky_decomposition(S2);
	size_t rnnz2 = 0;
	for (size_t i = 0; i < S2.rows; ++i) {
		uint32_t start = S2.row_start[i];
		uint32_t end = S2.row_start[i + 1];
		uint32_t nnzloc = 0;
		for (uint32_t k = start; k < end; ++k) {
			nnzloc += (S2.data[k] == 0 ? 0 : 1);
		}
		rnnz2 += nnzloc;
	}
	LOG_MSG("Actual NNZ2 : %zu (%.1f\%, %.1f * DOF * log_2(DOF))", rnnz2,
		(float)rnnz2 / S2.nnz * 100,
		(double)rnnz2 / (m2.vertex_count() * log2(m2.vertex_count())));
	spy(S2, 1024, "spy_dissected_cube.png");
	return (0);
}
