#include "array.h"

#define USE_FEM_MATRIX false
#if USE_FEM_MATRIX
	#include "fem_matrix.h"
#else
	#include "sparse_matrix.h"
#endif
#include "mesh.h"

struct PoissonSolver {
	PoissonSolver(const Mesh &m);
	const Mesh &m;
	size_t N;   // DoF
	double vol; // Surface(m), used for insuring zero mean to f and u
	TArray<double> f;
	TArray<double> u;
#if USE_FEM_MATRIX
	FEMatrix A; // Stiffness matrix
	FEMatrix M; // Mass matrix
#else
	CSRPattern P; // Pattern arrays
	CSRMatrix A;  // Stiffness matrix
	CSRMatrix M;  // Mass matrix
#endif
	TArray<double> r;  // current residue r = Mf - Su
	TArray<double> p;  // internal for cg
	TArray<double> Ap; // internal for cg

	bool inited;	  // Initialization computes first residue and error
	size_t iterate;	  // current iterate
	double b2;	  // ||Mf||^2
	double r2;	  // current ||r||^2
	bool converged;	  //
	double rel_error; // sqrt(r2 / b2)

	void clear_solution();
	void init_cg();
	void set_zero_mean(double *V);
	void do_iterate(size_t max_iter, double tol);
};
