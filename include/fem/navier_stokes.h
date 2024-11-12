#include "array.h"
#include "fem_matrix.h"
#include "mesh.h"

struct NavierStokesSolver {
	NavierStokesSolver(const Mesh &m);
	const Mesh &m;
	size_t N; // DoF
	double vol; // Surface(m), used for insuring zero mean to omega and psi

	TArray<double> omega;
	TArray<double> Momega;
	TArray<double> psi;
	FEMatrix S; // Stiffness matrix
	FEMatrix M; // Mass matrix
	TArray<double> r; // current residue r = Mf - Su
	TArray<double> p; // internal for cg
	TArray<double> Ap; // internal for cg

	bool inited; // Initialization computes first residue and error

	size_t iter_max = 5000;
	double tol = 1e-6;

	double t;

	void set_zero_mean(double *V);
	size_t compute_stream_function();
	void compute_transport(double *T);
	void time_step(double dt, double nu);
};
