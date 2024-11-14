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

/* A vector from its end points */
struct Vector vector(struct Vertex A, struct Vertex B)
{
	return {B.x - A.x, B.y - A.y, B.z - A.z};
}

double dot(struct Vector V, struct Vector W)
{
	return (V.x * W.x + V.y * W.y + V.z * W.z);
}

double norm(struct Vector V) { return sqrt(dot(V, V)); }

struct Vector cross(struct Vector V, struct Vector W)
{
	return {V.y * W.z - V.z * W.y, V.z * W.x - V.x * W.z,
		V.x * W.y - V.y * W.x};
}
/*****************************************************************************/

/******************************************************************************
 * Computes the product M * v when M is a SparseMatrix
 *****************************************************************************/
void matrix_vector_product(const struct SparseMatrix *M, const double *v,
			   double *Mv)
{
	for (int r = 0; r < M->rows; r++) {
		Mv[r] = 0;
	}
	for (int k = 0; k < M->nnz; k++) {
		int i = M->coeffs[k].i;
		int j = M->coeffs[k].j;
		assert(i < M->rows);
		assert(j < M->cols);
		double Mij = M->coeffs[k].val;
		Mv[i] += Mij * v[j];
	}
}

/******************************************************************************
 * Builds the P1 stiffness and mass matrices of a given mesh.
 * We do not try to assemble different elements together here for simplicity.
 * Both matrices M and S will therefore have 9 * number of triangles.
 * We derived the formulas in the first lecture.
 *****************************************************************************/
void build_fem_matrices(const struct Mesh *m, struct SparseMatrix *S,
			struct SparseMatrix *M)
{
	int N = m->vtx_count;
	S->rows = S->cols = M->rows = M->cols = m->vtx_count;
	/* We do not try to assemble, so 9 coeffs per triangle */
	S->nnz = M->nnz = 9 * m->tri_count;
	S->coeffs = (struct Coeff *)malloc(S->nnz * sizeof(struct Coeff));
	M->coeffs = (struct Coeff *)malloc(M->nnz * sizeof(struct Coeff));

	for (int i = 0; i < m->tri_count; i++) {
		int a = m->triangles[i].a;
		int b = m->triangles[i].b;
		int c = m->triangles[i].c;
		assert(a < N);
		assert(b < N);
		assert(c < N);
		struct Vertex A = m->vertices[a];
		struct Vertex B = m->vertices[b];
		struct Vertex C = m->vertices[c];

		struct Vector AB = vector(A, B);
		struct Vector BC = vector(B, C);
		struct Vector CA = vector(C, A);

		struct Vector CAxAB = cross(CA, AB);
		double area = 0.5 * norm(CAxAB);

		struct Coeff *m = &M->coeffs[9 * i];
		m[0] = {a, a, area / 6};
		m[1] = {b, b, area / 6};
		m[2] = {c, c, area / 6};
		m[3] = {a, b, area / 12};
		m[4] = {b, a, area / 12};
		m[5] = {a, c, area / 12};
		m[6] = {c, a, area / 12};
		m[7] = {b, c, area / 12};
		m[8] = {c, b, area / 12};

		struct Coeff *s = &S->coeffs[9 * i];
		double r = 1. / (4 * area);
		s[0] = {a, a, dot(BC, BC) * r};
		s[1] = {b, b, dot(CA, CA) * r};
		s[2] = {c, c, dot(AB, AB) * r};
		s[3] = {a, b, dot(BC, CA) * r};
		s[4] = {b, a, dot(BC, CA) * r};
		s[5] = {a, c, dot(AB, BC) * r};
		s[6] = {c, a, dot(AB, BC) * r};
		s[7] = {b, c, dot(AB, CA) * r};
		s[8] = {c, b, dot(AB, CA) * r};
	}
}

/******************************************************************************
 * Routines for elementary linear algebra in arbitrary (large) dimensions
 *****************************************************************************/

/* Create an (unitialized) array of N double precision floating point values */
double *array(int N) { return (double *)malloc(N * sizeof(double)); }

/* Vector product between two vectors in dim N */
double blas_dot(const double *A, const double *B, int N)
{
	double res = 0.0;
	for (int i = 0; i < N; ++i) {
		res += A[i] * B[i];
	}
	return res;
}

/* aX + bY -> Y  (axpby reads as aX plus bY)
 * a and b are scalar, X and Y are vectors in dim N
 */
void blas_axpby(double a, const double *X, double b, double *Y, int N)
{
	for (int i = 0; i < N; i++) {
		Y[i] = a * X[i] + b * Y[i];
	}
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
			  double *U, int N)
{
	/* Since S + M is symmetric and positive definite we can solve
	 * the system by the gradient descent method. This is by far
	 * not the best method for ill conditionned matrices, but the point
	 * here is not (yet) optimality.
	 *
	 * We need to solve AU = B with A := (S + M).
	 * The iterates are constructed as follows : given an approximation
	 * U_k we compute the residue
	 *
	 *          r_k = B - AU_k
	 *
	 * Then we follow the negative gradient and build
	 *
	 *                           <r_k, r_k>
	 *          U_{k+1} = U_k +  ----------- r_k =: U_k + alpha_k r_k
	 *                           <Ar_k, r_k>
	 *
	 * until the residue becomes smaller to a given threshold.
	 *
	 * Note : r_{k + 1} = B - A(U_k + alpha_k r_k) = r_k - alpha_k Ar_k
	 *        That will save us one matrix vector product.
	 */

	/* Set U_0 to the zero vector */
	memset(U, 0, N * sizeof(double));

	/* r_0 = B - (M+S)U_0 = B */
	double *r = array(N);
	memcpy(r, B, N * sizeof(double));

	/* l^2 (squared) norm of the residue */
	double error2 = blas_dot(r, r, N);

	/* Temporaries */
	double *Mr = array(N);
	double *Ar = array(N);

	double tol2 = 1e-6;
	int iterate = 0;
	while (error2 > tol2) {
		iterate++;
		/* Compute Ar_k */
		matrix_vector_product(S, r, Ar);
		matrix_vector_product(M, r, Mr);
		blas_axpby(1, Mr, 1, Ar, N);
		assert(blas_dot(Ar, r, N) > 0);
		/* Compute alpha_k */
		double alpha = blas_dot(r, r, N) / blas_dot(Ar, r, N);

		/* Update U */
		blas_axpby(alpha, r, 1, U, N);

		/* Update r */
		blas_axpby(-alpha, Ar, 1, r, N);

		/* Update error2 */
		error2 = blas_dot(r, r, N);
	}
	/* Release system memory */
	free(Ar);
	free(Mr);
	free(r);
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

	struct SparseMatrix M = {0, 0, 0, NULL};
	struct SparseMatrix S = {0, 0, 0, NULL};
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
int build_cube_vertices(struct Vertex *vert, int N)
{
	int V = N + 1;
	int nvf = V * V;
	int v = 0;
	
	for (int row = 0; row < V; row++) {
		for (int col = 0; col < V; col++) {
			vert[0 * nvf + v].x = col;
			vert[0 * nvf + v].y = 0;
			vert[0 * nvf + v].z = row;

			vert[1 * nvf + v].x = N;
			vert[1 * nvf + v].y = col;
			vert[1 * nvf + v].z = row;

			vert[2 * nvf + v].x = N - col;
			vert[2 * nvf + v].y = N;
			vert[2 * nvf + v].z = row;

			vert[3 * nvf + v].x = 0;
			vert[3 * nvf + v].y = N - col;
			vert[3 * nvf + v].z = row;

			vert[4 * nvf + v].x = col;
			vert[4 * nvf + v].y = N - row;
			vert[4 * nvf + v].z = 0;

			vert[5 * nvf + v].x = col;
			vert[5 * nvf + v].y = row;
			vert[5 * nvf + v].z = N;

			v++;
		}
	}
	return 6 * nvf;
}

int build_cube_triangles(struct Triangle *tri, int N)
{
	int V = N + 1;
	int t = 0;
	for (int face = 0; face < 6; face++) {
		for (int row = 0; row < N; row++) {
			for (int col = 0; col < N; col++) {
				int v = face * V * V + row * V + col;
				tri[t++] = {v, v + 1, v + 1 + V};
				tri[t++] = {v, v + 1 + V, v + V};
			}
		}
	}
	assert(t == 12 * N * N);
	return t;
}

int dedup_mesh_vertices(struct Mesh *m)
{
	int vtx_count = 0;
	int V = m->vtx_count;
	int *remap = (int *)malloc(V * sizeof(int));
	/* TODO replace that inefficient linear search ! */
	for (int i = 0; i < V; i++) {
		bool dup = false;
		struct Vertex v = m->vertices[i];
		for (int j = 0; j < i; j++) {
			struct Vertex vv = m->vertices[j];
			if (v.x == vv.x && v.y == vv.y && v.z == vv.z) {
				dup = true;
				remap[i] = remap[j];
				break;
			}
		}
		if (!dup) {
			remap[i] = vtx_count;
			vtx_count++;
		}
	}
	/* Remap vertices */
	for (int i = 0; i < m->vtx_count; i++) {
		m->vertices[remap[i]] = m->vertices[i];
	}
	/* Remap triangle indices */
	for (int i = 0; i < m->tri_count; i++) {
		struct Triangle *T = &m->triangles[i];
		T->a = remap[T->a];
		assert(T->a < vtx_count);
		T->b = remap[T->b];
		assert(T->b < vtx_count);
		T->c = remap[T->c];
		assert(T->c < vtx_count);
	}
	free(remap);
	return vtx_count;
}

void build_cube_mesh(struct Mesh *m, int N)
{
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
void send_cube_to_sphere(struct Vertex *vert, int vtx_count)
{
	for (int i = 0; i < vtx_count; i++) {
		struct Vertex *v = &vert[i];
		double norm = sqrt(v->x * v->x + v->y * v->y + v->z * v->z);
		v->x /= norm;
		v->y /= norm;
		v->z /= norm;
	}
}
/*****************************************************************************/
