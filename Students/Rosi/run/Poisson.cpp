#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hash_table.h"
#include "tiny_blas.h"
#include "my_mesh.h"
#include "array.h"
#include "P1.h"
#include "conj_gradient.h"

#define CRS 1

#ifndef CRS
#include "fem_matrix.h"
#else
#include "sparse_matrix.h"
#endif

double f(double x, double y, double z)
{
	(void)z; 
	return x * x - y * y;
}


int main(int argc, char **argv)
{
	if (argc < 2)
		return EXIT_FAILURE;

	struct Mesh m;
	build_cube_mesh(&m, atoi(argv[1]));
	send_cube_to_sphere(m.vertices, m.vtx_count);
	int N = m.vtx_count;

	printf("Number of DOF : %d\n", N);
	
	#ifndef CRS
	struct FEMatrix M;
	struct FEMatrix S;
	build_P1_mass_matrix(m, M);
    build_P1_stiffness_matrix(m, S); 
	#else
	CSRMatrix M;
	CSRMatrix S;
	struct CSRPattern P;

	// I don't know why it gives me these errors but it works

	build_P1_CSRPattern(m, P);
	build_P1_mass_matrix(m, P, M);
	build_P1_stiffness_matrix(m, P, S);

	#endif
	
	TArray<double> U(N,0);
	TArray<double> F(N);
	

	/* Fill F */
	
	for (int i = 0; i < N; i++) {
		Vec3d v = m.vertices[i];
		F[i] = f(v.x, v.y, v.z);
	}

	S.mvp(F.data, U.data);

	/* Fill B = MF */
	TArray<double> B(N);
	M.mvp(F.data, B.data);

	/* Solve (S + M)U = B */
    double error;
	
	int iter = conj_gradient(&S, &M, &B, &U, &error, N);
	printf("System solved in %d iterations.\n", iter);
    printf("Absolute value of residue (l^2): %f\n", error);

	printf("Integrity check :\n");
	printf("-----------------\n");
	for (int i = 0; i < 8; i++) {
		if (F[i] != 0) {
			printf("Ratio U/F : %f\n", U[i] / F[i]);
		}
	}

	return (EXIT_SUCCESS);
}