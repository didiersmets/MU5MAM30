#include <string.h>

#include "P1.h"
#include "cholesky.h"
#include "cube.h"
#include "elimination_tree.h"
#include "logging.h"
#include "mesh.h"
#include "sparse_matrix.h"

int main(int argc, char **argv)
{
	log_init(0);

	Mesh m1, m2;

	if (argc > 1) {
		uint32_t subdiv = atoi(argv[1]);
		load_cube(m1, subdiv);
		LOG_MSG("Loaded standard cube.");
		load_cube_nested_dissect(m2, subdiv);
		LOG_MSG("Loaded dissected cube.");
		LOG_MSG("Vertices : %zu. Triangles : %zu", m2.vertex_count(),
			m2.triangle_count());
	} else {
		return (EXIT_FAILURE);
	}

#if 0
	SKLPattern P1;
	SKLMatrix S1;
	build_P1_SKLPattern(m1, P1);
	build_P1_stiffness_matrix(m1, P1, S1);
	size_t rnnz = 0;
	for (size_t i = 0; i < S1.nnz; ++i) {
		rnnz += (S1.data[i] == 0 ? 0 : 1);
	}
	LOG_MSG("Stiffeness NNZ : %10zu", rnnz);
	LOG_MSG("       SKL NNZ : %10zu (ratio = %5.1f)", S1.nnz,
		(float)S1.nnz / rnnz);
	in_place_cholesky_decomposition(S1);
	size_t rnnz1 = 0;
	for (size_t i = 0; i < S1.nnz; ++i) {
		rnnz1 += (S1.data[i] == 0 ? 0 : 1);
	}
	LOG_MSG("      Chol NNZ : %10zu (ratio = %5.1f)", rnnz1,
		(float)rnnz1 / rnnz);
	spy(S1, 1024, "spy_normal_cube.png");

	SKLPattern P2;
	SKLMatrix S2;
	build_P1_SKLPattern(m2, P2);
	build_P1_stiffness_matrix(m2, P2, S2);
	in_place_cholesky_decomposition(S2);
	size_t rnnz2 = 0;
	for (size_t i = 0; i < S2.nnz; ++i) {
		rnnz2 += (S2.data[i] == 0 ? 0 : 1);
	}
	LOG_MSG(" Ord. Chol NNZ : %10zu (ratio = %5.1f)", rnnz2,
		(float)rnnz2 / rnnz);
	spy(S2, 1024, "spy_dissected_cube.png");
#endif
	CSRPattern P;
	CSRMatrix S;
	TArray<uint32_t> parent;
	size_t fill_in;

	build_P1_CSRPattern(m1, P);
	build_P1_stiffness_matrix(m1, P, S);
	LOG_MSG("Starting NNZ : %zu", S.nnz);

	build_elimination_tree(S, parent);
	fill_in = compute_fill_in(S, parent);
	LOG_MSG("Natural ordering fill-in  : %10zu ( = %4.1f * NNZ^(3/2))",
		fill_in, (float)fill_in / (pow(S.nnz, 1.5)));

	build_P1_CSRPattern(m2, P);
	build_P1_stiffness_matrix(m2, P, S);
	build_elimination_tree(S, parent);
	fill_in = compute_fill_in(S, parent);
	LOG_MSG("Nested bissection fill-in : %10zu ( = %4.1f * NNZ * "
		"log2(NNZ))",
		fill_in, (float)fill_in / (S.nnz * log2(S.nnz)));

	return (0);
}
