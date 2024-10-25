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
struct Vector vector(struct Vertex A, struct Vertex B){
	return {B.x - A.x, B.y - A.y, B.z - A.z};
}

double dot(struct Vector V, struct Vector W){
	return V.x*W.x + V.y*W.y + V.z*W.z;
}

double norm(struct Vector V) { return sqrt(dot(V, V)); };

struct Vector cross(struct Vector V, struct Vector W){
	return {V.y*W.z - V.z*W.y, V.z*W.x - V.x*W.z, V.x*W.y - V.y*W.x};
}
/*****************************************************************************/

/******************************************************************************
 * Computes the product M * v when M is a SparseMatrix
 *****************************************************************************/
void matrix_vector_product(const struct SparseMatrix *M, const double *v,
			   double *Mv){
    for(int i = 0; i < M->rows; i++) Mv[i] = 0;

	for (int k = 0; k < M->nnz; k++) 
		Mv[M->coeffs[k].i] += M->coeffs[k].val * v[M->coeffs[k].j];
	return;			
}

/******************************************************************************
 * Builds the P1 stiffness and mass matrices of a given mesh.
 * We do not try to assemble different elements together here for simplicity.
 * Both matrices M and S will therefore have 9 * number of triangles.
 * We derived the formulas in the first lecture.
 *****************************************************************************/
void build_fem_matrices(const struct Mesh *m, struct SparseMatrix *S,
			struct SparseMatrix *M){
	int N = m->vtx_count;
    S->cols = S->rows = M->cols = M->rows = N;
    S->nnz = M->nnz = 9*m->tri_count;
    S->coeffs = (struct Coeff *) malloc(S->nnz * sizeof(struct Coeff));
    M->coeffs = (struct Coeff *) malloc(M->nnz * sizeof(struct Coeff));
    

	for(int t = 0; t < m->tri_count; t++){
        int a = m->triangles[t].a;
        int b = m->triangles[t].b;
        int c = m->triangles[t].c;

		struct Vertex A = m->vertices[a];
		struct Vertex B = m->vertices[b];
		struct Vertex C = m->vertices[c];

		struct Vector AB = vector(A, B);
        struct Vector BC = vector(B, C);
        struct Vector CA = vector(C, A);

        double area = 0.5 * norm(cross(AB, CA));
        
        struct Coeff * mass = &M->coeffs[9*t];
        mass[0] = {a, a, area/6};
        mass[1] = {b, b, area/6};
        mass[2] = {c, c, area/6};
        mass[3] = {a, b, area/12};
        mass[4] = {a, c, area/12};
        mass[5] = {b, a, area/12};
        mass[6] = {c, a, area/12};
        mass[7] = {c, b, area/12};
        mass[8] = {b, c, area/12};

        struct Coeff * stiff = &S->coeffs[9*t];
        double mult = 1. / (4*area);
        stiff[0] = {a,a,dot(BC,BC) * mult};
        stiff[1] = {b,b,dot(CA,CA) * mult};
        stiff[2] = {c,c,dot(AB,AB) * mult};
		stiff[3] = {a,b,dot(BC,BC) * mult};
        stiff[4] = {b,a,dot(CA,BC) * mult};
        stiff[5] = {a,c,dot(CA,BC) * mult};
        stiff[6] = {c,a,dot(AB,BC) * mult};
        stiff[7] = {b,c,dot(AB,CA) * mult};
        stiff[8] = {c,b,dot(AB,CA) * mult};
	}
	return;
}

/******************************************************************************
 * Routines for elementary linear algebra in arbitrary (large) dimensions
 *****************************************************************************/

/* Create an (unitialized) array of N double precision floating point values */
double *array(int N) { 
    return (double *)malloc(N * sizeof(double)); 
}

/* Vector product between two vectors in dim N */
double blas_dot(const double *A, const double *B, int N){
    double res = 0;
    for(int i = 0; i < N; i++)
        res += A[i]*B[i];
    return res;
}

/* aX + bY -> Y  (axpby reads as aX plus bY)
 * a and b are scalar, X and Y are vectors in dim N
 */
void blas_axpby(double a, const double *X, double b, double *Y, int N){
    for (int i = 0; i < N; i++)
        Y[i] = a*X[i] + b*Y[i];
    return;
}

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
			  double *U,double *error, int N){
    //initial residue
    double *r = array(N);
    double *Mr = array(N);
    double *Ar = array(N);
    matrix_vector_product(S,U,r);
    matrix_vector_product(M,U,Mr);
    blas_axpby(1,Mr,1,r,N);
    blas_axpby(1,B,-1,r,N);

    int iterate = 0;
    int iter_max = 1000;
    double tol = 1e-6;
    double tol2 = tol*tol;
    double error2 = blas_dot(r,r,N);
    while(error2 > tol2 && iterate < iter_max){

        matrix_vector_product(S,r,Ar);
        matrix_vector_product(M,r,Mr);
        blas_axpby(1,Mr,1,Ar,N);
        double alpha = error2/blas_dot(Ar,r,N);
        //Update U
        blas_axpby(alpha,r,1,U,N);
    
        //Update r (note r{k+1} = r{k} - aplha*(Ar{k}))
        blas_axpby(-alpha,Ar,1,r,N);

        //Update error
        error2 = blas_dot(r,r,N);

        iterate++;
    }
    free(Mr);
    free(Ar);
    free(r);

    *error = sqrt(error2);
    
    return iterate;

}

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
    double error;
	double *U = array(N);
	int iter = gradient_system_solve(&S, &M, B, U, &error, N);
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

/******************************************************************************
 * Building a cube surface mesh. N is the number of subdivisions per side.
 * ***************************************************************************/
int build_cube_vertices(struct Vertex *vert, int N){
    (void) vert;
    (void) N;
    return 0;
}

int build_cube_triangles(struct Triangle *tri, int N){
    (void) tri;
    (void) N;
    return 0;
}

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
void send_cube_to_sphere(struct Vertex *vert, int vtx_count){
    (void) vert;
    (void) vtx_count;
}
/*****************************************************************************/
