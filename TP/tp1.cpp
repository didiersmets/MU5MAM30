#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The goal of this TP is to get acquainted with C (not yet C++)
 * and at the same time solve the equation
 *
 *    - \Delta u + u = f
 *
 * on (e.g.) the sphere S^2. The sphere is handy for testing because
 * "we" know explicit solutions (besides constants) for some particular
 * cases of f (which ones ?), but process would work on an arbitrary surface
 * mesh.
 *
 * Detailed instructions shall be given on the blackboard prior to the class.
 * Depending on their background in both programming and the FEM, students will
 * have the choice to either :
 * 1) follow me coding/explaining live, and ask questions
 * 2) fill the gaps by themselves, following the proposed structure
 * 3) provide some conceptual (algorithmmic/performance/robustness) improvements
 * or a mix of these...
 */

/* The C structures we define for our needs */

/* A Vertex is simply a point in R^3 */
struct Vertex {
	double x;
	double y;
	double z;
};

/* A triangle refers to three vertices by their index */
struct Triangle {
	int a;
	int b;
	int c;
};

/* A mesh is an array of vertices, and an array of triangles
 * built over these vertices
 */
struct Mesh {
	int vtx_count;
	int tri_count;
	struct Vertex *vertices;
	struct Triangle *triangles;
};

/* A coefficient of a sparse matrix */
struct Coeff {
	int i;
	int j;
	double val;
};

/* A Sparse matrix is an (order independent) array of coeffs,
 * the non zero entries of the matrix.
 * That for is called a COO sparse matrixe (COO for coordinates).
 * It is the simplest form and not the most efficient.
 * In the main repo we use a (probably) better approach.
 */
struct SparseMatrix {
	int rows;
	int cols;
	int nnz;
	struct Coeff *coeffs;
};

/* A Vector in R^3. Similar to Vertex, but as in affine geometry we may
 * wish to distinguish vectors and points.
 */
struct Vector {
	double x;
	double y;
	double z;
};

/******************************************************************************
 * Vectors in R^3
 ******************************************************************************/

/* Returns a vector from its end points */
struct Vector vector(struct Vertex A, struct Vertex B);

double dot(struct Vector V, struct Vector W);

double norm(struct Vector V) { return sqrt(dot(V, V)); };

struct Vector cross(struct Vector V, struct Vector W);
/*****************************************************************************/

/******************************************************************************
 * Computes the product M * v when M is a SparseMatrix
 *****************************************************************************/
void matrix_vector_product(const struct SparseMatrix *M, const double *v,
			   double *Mv);

/******************************************************************************
 * Builds the P1 stiffness and mass matrices of a given mesh.
 * We do not try to assemble different elements together here for simplicity.
 * Both matrices M and S will therefore have 9 * number of triangles.
 * We derived the formulas in the first lecture.
 *****************************************************************************/
void build_fem_matrices(const struct Mesh *m, struct SparseMatrix *S,
			struct SparseMatrix *M);

/******************************************************************************
 * Routines for elementary linear algebra in arbitrary (large) dimensions
 *****************************************************************************/

/* Create an (unitialized) array of N double precision floating point values */
double *array(int N) { return (double *)malloc(N * sizeof(double)); }

/* Vector product between two vectors in dim N */
double blas_dot(const double *A, const double *B, int N);

/* aX + bY -> Y  (axpby reads as aX plus bY)
 * a and b are scalar, X and Y are vectors in dim N
 */
void blas_axpby(double a, const double *X, double b, double *Y, int N);

/******************************************************************************
 * Solving AU=B where A is SPD of size NxN using steepest descent method.
 * Find it back on a sheet of paper, not on Google !
 * One minimizes the functional 1/2 <AU,U> - <B,U>.
 * The minor peculiarity here is that A = S + M and we do not wish to
 * add these two sparse matrices up-front but simply compute AU as SU + MU
 * wherever needed.
 *****************************************************************************/
int gradient_system_solve(const struct SparseMatrix *S,
			  const struct SparseMatrix *M, const double *B,
			  double *U, int N);

/******************************************************************************
 * Let's choose our right hand side f of -\Delta u + u = f
 *****************************************************************************/
double f(double x, double y, double z)
{
	(void)z; /* Avoids compiler warning about unused variable */
	return x * x - y * y;
}

/******************************************************************************
 * Mesh construction routine declared here, defined later below main routine.
 *****************************************************************************/
void build_cube_mesh(struct Mesh *m, int N);
void send_cube_to_sphere(struct Vertex *vert, int vtx_count);

/******************************************************************************
 * Main routine
 *****************************************************************************/
int main(int argc, char **argv)
{
	if (argc < 2)
		return EXIT_FAILURE;

	struct Mesh m;
	build_cube_mesh(&m, atoi(argv[1]));
	send_cube_to_sphere(m.vertices, m.vtx_count);
	int N = m.vtx_count;
	printf("Number of DOF : %d\n", N);

	struct SparseMatrix M;
	struct SparseMatrix S;
	build_fem_matrices(&m, &S, &M);

	/* Fill F */
	double *F = array(N);
	for (int i = 0; i < N; i++) {
		struct Vertex v = m.vertices[i];
		F[i] = f(v.x, v.y, v.z);
	}
	/* Fill B = MF */
	double *B = array(N);
	matrix_vector_product(&M, F, B);

	/* Solve (S + M)U = B */
	double *U = array(N);
	int iter = gradient_system_solve(&S, &M, B, U, N);
	printf("System solved in %d iterations.\n", iter);

	printf("Integrity check :\n");
	printf("-----------------\n");
	for (int i = 0; i < 8; i++) {
		if (F[i] != 0) {
			printf("Ratio U/F : %f\n", U[i] / F[i]);
		}
	}

	return (EXIT_SUCCESS);
}

/******************************************************************************
 * Building a cube surface mesh. N is the number of subdivisions per side.
 * ***************************************************************************/
int build_cube_vertices(struct Vertex *vert, int N);

int build_cube_triangles(struct Triangle *tri, int N);

int dedup_mesh_vertices(struct Mesh *m);

void build_cube_mesh(struct Mesh *m, int N)
{
	/* Number of vertices per side = number of divisions + 1 */
	int V = N + 1;

	/* We allocate for 6 * V^2 vertices */
	int max_vert = 6 * V * V;
	m->vertices = (struct Vertex *)malloc(max_vert * sizeof(struct Vertex));
	m->vtx_count = 0;

	/* We allocate for 12 * N^2 triangles */
	int tri_count = 12 * N * N;
	m->triangles =
	    (struct Triangle *)malloc(tri_count * sizeof(struct Triangle));
	m->tri_count = 0;

	/* We fill the vertices and then the faces */
	m->vtx_count = build_cube_vertices(m->vertices, N);
	m->tri_count = build_cube_triangles(m->triangles, N);

	/* We fix-up vertex duplication */
	m->vtx_count = dedup_mesh_vertices(m);
	assert(m->vtx_count == 6 * V * V - 12 * V + 8);

	/* Rescale to unit cube centered at the origin */
	for (int i = 0; i < m->vtx_count; ++i) {
		struct Vertex *v = &m->vertices[i];
		v->x = 2 * v->x / N - 1;
		v->y = 2 * v->y / N - 1;
		v->z = 2 * v->z / N - 1;
	}
}

/******************************************************************************
 * The so-called spherical cube, built by simply normalizing all vertices of
 * the cube mesh so that they end up in S^2
 *****************************************************************************/
void send_cube_to_sphere(struct Vertex *vert, int vtx_count);
/*****************************************************************************/
