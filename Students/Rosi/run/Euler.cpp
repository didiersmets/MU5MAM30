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
#include "cholesky.h"
#include "skyline.h"
#include "sparse_matrix.h"

int main(int argc, char **argv)
{
	if (argc < 2)
		return EXIT_FAILURE;

	struct Mesh m;
	build_cube_mesh(&m, atoi(argv[1]));
	send_cube_to_sphere(m.vertices, m.vtx_count);
	int N = m.vtx_count;

	printf("Number of DOF : %d\n", N);

	CSRMatrix M;
	SkylineMatrix S;
	CSRPattern P;

	// I don't know why it gives me these errors but it works

	build_P1_CSRPattern(m, P);
	build_P1_mass_matrix(m, P, M);
	build_P1_stiffness_matrix(m, S);
	
	TArray<double> U(N,0);
	TArray<double> F(N,2);

	//S.mvp(F.data, U.data);

	/* Fill B = MF */
	TArray<double> B(N);
	M.mvp(F.data, B.data);

	for(int i = 0; i < 5; i++)
		printf("%f\n",B[i]);

    double error;

    double TF = 2.0;
    double timestep = 0.2;
	
	TArray<double> errors(TF/timestep,0.0);
    TArray<double> omega(N,1.0);
    TArray<double> psi(N,0.0);
	
    for (double dt = 0.; dt < TF; dt += timestep){
        
		TArray<double> T(N,0.0);
        for (size_t t = 0; t < m.tri_count; t++) {
        int a = m.triangles[t].x;
        int b = m.triangles[t].y;
        int c = m.triangles[t].z;
        assert(a < N && b < N && c < N);
        double sum_omega = omega[a] + omega[b] + omega[c];

        if (t == a)
            T[t] += psi[b]*sum_omega - psi[c]*sum_omega;
        if (t == b)
            T[t] += psi[c]*sum_omega - psi[a]*sum_omega;
        if (t == c)
            T[t] += psi[a]*sum_omega - psi[b]*sum_omega;
        }
        
        // Solve M*omega(k+1) = M*(omega) + dt*T + dt*M*F
		printf("Solving for omega\n");
		TArray<double> Momega(N,0);
        M.mvp(omega.data, Momega.data);
        blas_axpby(dt, T.data, dt, B.data, N);
		blas_axpby(1, Momega.data, 1, B.data, N);

		for(int i = 0; i < 5; i++)
			printf("%f\n",B[i]);
        TArray<double> omega_next(N,0);

		double error;

	    int iter = conj_gradient(&M, &B, &omega, &error, N);
		errors[dt/timestep] = error;
	    printf("Time %f. System solved in %d iterations. Error: %f\n", dt, iter, error);
		printf("Norm of omega: %f\n",blas_dot(omega.data,omega.data,N));
        // Solve S*psi(k+1) = -M*omega(k+1)
		//memcpy(omega.data,omega_next.data,N);

		blas_axpby(-1, omega.data, 0, B.data, N);
		for(int i = 0; i < N; i++)
			printf("%f\n",B[i]);
        M.mvp(omega.data, B.data);
        TArray<double> psi_next(N,0);
		SkylineMatrix L;
		// Compute the Cholesky Matrixx
		//Cholesky(&S,&L);
		//CholeskySolve(&L,&B,&psi_next);

		iter = conj_gradient(&S, &B, &psi_next, &error, N);
		printf("System solved in %d iterations. Error:%f\n", iter, error);

		memcpy(psi.data,psi_next.data,N);
		//print the norm of psi
		double norm = blas_dot(psi.data,psi.data,N);
		
    }

	return (EXIT_SUCCESS);
}