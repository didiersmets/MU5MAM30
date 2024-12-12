
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

	uint32_t subdiv;

	if (argc > 1) {
		subdiv = atoi(argv[1]);
	} else {
		return (EXIT_FAILURE);
	}

	/* Standard ordering */
	if (subdiv <= 128) {
		Mesh m1;
		load_cube(m1, subdiv);
		LOG_MSG("Loaded standard cube.");
		LOG_MSG("Vertices: %10zu Triangles : %zu", m1.vertex_count(),
			m1.triangle_count());
		SKLPattern P1;
		SKLMatrix S1;
		build_P1_SKLPattern(m1, P1);

		build_P1_stiffness_matrix(m1, P1, S1);
		size_t snnz = 0;
		for (size_t i = 0; i < S1.nnz; ++i) {
			snnz += (S1.data[i] == 0 ? 0 : 1);
		}
		LOG_MSG("S   NNZ : %10zu", snnz);
		LOG_MSG("SKL RAW : %10zu (ratio = %5.1f)", S1.nnz,
			(float)S1.nnz / snnz);
		in_place_cholesky_factorization(S1);
		size_t lnnz = 0;
		for (size_t i = 0; i < S1.nnz; ++i) {
			lnnz += (S1.data[i] == 0 ? 0 : 1);
		}
		LOG_MSG("L   NNZ : %10zu (ratio = %5.1f)\n", lnnz,
			(float)lnnz / snnz);
		spy(S1, 1024, "./data/spy_normal_cube.png");
	} else {
		LOG_MSG("Standard cube ordering skipped for subdiv > 128.\n");
	}

	/* Nested dissection ordering */
	{
		Mesh m2;
		load_cube_nested_dissect(m2, subdiv);
		LOG_MSG("Loaded dissected cube.");
		LOG_MSG("Vertices: %10zu Triangles : %zu", m2.vertex_count(),
			m2.triangle_count());

		CSRPattern P2, P3;
		CSRMatrix S2, S3;
		build_P1_CSRPattern(m2, P2);
		build_P1_stiffness_matrix(m2, P2, S2);
		LOG_MSG("S   NNZ : %10zu", S2.nnz);
		csr_build_cholesky_pattern(P2, P3);
		LOG_MSG("L   NNZ : %10zu (ratio = %5.1f, count = %4.1f * NNZ * "
			"log2(NNZ))",
			P3.nnz, (float)P3.nnz / S2.nnz,
			(float)P3.nnz / (S2.nnz * log2(S2.nnz)));
		size_t phi = 0;
		for (size_t i = 0; i < P3.rows; ++i) {
			size_t eta = P3.row_start[i + 1] - P3.row_start[i];
			phi += eta * eta;
		}
		LOG_MSG("L   PHI : %10zu (%.2f GFlops)", phi,
			double(phi) / 1e9);
		csr_cholesky_factorization(S2, P3, S3);
		LOG_MSG("Cholesky finished");

		// spy(P3, 1024, "./data/spy_dissected_cube.png");
	}

	return (0);
}
