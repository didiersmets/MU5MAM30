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
	struct Vertex* vertices;
	struct Triangle* triangles;
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
	struct Coeff* coeffs;
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
struct Vector vector(struct Vertex A, struct Vertex B) {
	struct Vector res;
	res.x = B.x - A.x;
	res.y = B.y - A.y;
	res.z = B.z - A.z;
	return res;
}

double dot(struct Vector V, struct Vector W) {
	return V.x * W.x + V.y * W.y + V.z + W.z;
}

double norm(struct Vector V) { 
	return sqrt(dot(V, V)); 
}

struct Vector cross(struct Vector V, struct Vector W) {
	struct Vector res;
	res.x = V.y * W.z - V.z * W.y;
	res.y = V.z * W.x - V.x * W.z;
	res.z = V.x * W.y - V.y * W.x;
	return res;
}
/*****************************************************************************/

/******************************************************************************
 * Computes the product M * v when M is a SparseMatrix
 *****************************************************************************/
void matrix_vector_product(const struct SparseMatrix* M, const double* v, double* Mv) {
	for (int i = 0; i <M->rows ; i++) {
		Mv[i] = 0;

	}
	for (int k = 0; k < M->nnz; k++) {
		struct Coeff* c = &M->coeffs[k];
		Mv[c->i] += c->val * v[c->j];

	}
}

/******************************************************************************
 * Builds the P1 stiffness and mass matrices of a given mesh.
 * We do not try to assemble different elements together here for simplicity.
 * Both matrices M and S will therefore have 9 * number of triangles.
 * We derived the formulas in the first lecture.
 *****************************************************************************/
void build_fem_matrices(const struct Mesh* m, struct SparseMatrix* S, struct SparseMatrix* M) {
	int N=m->vtx_count;
	S->cols = S->rows = M->cols = M->rows = N;
	S->nnz = M->nnz = 9 * m->tri_count;
	S->coeffs = (struct Coeff*)malloc(S->nnz * sizeof(struct Coeff));
	M->coeffs = (struct Coeff*)malloc(M->nnz * sizeof(struct Coeff));
	for (int t = 0; t < m->tri_count; t++) {
		int a = m->triangles[t].a;
		int b = m->triangles[t].b;
		int c = m->triangles[t].c;
		struct Vertex A = m->vertices[a];
		struct Vertex B = m->vertices[b];
		struct Vertex C = m->vertices[c];
		struct Vector AB = vector(A,B);
		struct Vector BC = vector(B,C);
		struct Vector CA = vector(C,A);

		double area = 0.5 * norm(cross(AB, BC));
		struct Coeff *mass = &M->coeffs[9 * t];
		mass[0] = {a,a,area/6};
		mass[1] = { b,b,area / 6 };
		mass[2] = { c,c,area / 6 };
		mass[3] = { a,b,area / 12 };
		mass[4] = { b,a,area / 12 };
		mass[5] = { a,c,area / 12 };
		mass[6] = { c,a,area / 12 };
		mass[7] = { b,c,area / 12 };
		mass[8] = { c,b,area / 12 };
		struct Coeff* stiff = &S->coeffs[9 * t];
		double mult = 1. / (4 * area);
		stiff[0] = { a,a,dot(BC,BC) * mult };
		stiff[1] = { b,b,dot(CA,CA) * mult };
		stiff[2] = { c,c,dot(AB,AB) * mult };
		stiff[3] = { a,b,dot(CA,BC) * mult };
		stiff[4] = { b,a,dot(CA,BC) * mult };
		stiff[5] = { a,c,dot(AB,BC) * mult };
		stiff[6] = { c,a,dot(AB,BC) * mult };
		stiff[7] = { b,c,dot(AB,CA) * mult };
		stiff[8] = { c,b,dot(AB,CA) * mult };


	}
}

/******************************************************************************
 * Routines for elementary linear algebra in arbitrary (large) dimensions
 *****************************************************************************/

 /* Create an (unitialized) array of N double precision floating point values */
double* array(int N) { 
	return (double *)malloc(N * sizeof(double)); 
}

/* Vector product between two vectors in dim N */
double blas_dot(const double* A, const double* B, int N) {
	double res = 0;
	for (int i = 0; i < N; i++) {
		res += A[i] * B[i];
	}
	return res;
}

/* aX + bY -> Y  (axpby reads as aX plus bY)
 * a and b are scalar, X and Y are vectors in dim N
 */
void blas_axpby(double a, const double* X, double b, double* Y, int N) {
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
int gradient_system_solve(const struct SparseMatrix* S, const struct SparseMatrix* M, const double* B, double* U,double* error, int N) {
	//initial residue
	double* r = array(N);
	double* Ar = array(N);
	double* Mr = array(N);
	matrix_vector_product(S,U,r);
	printf("r=%f\n", r[0]);
	matrix_vector_product(M, U, Mr);
	printf("Mr=%f\n", Mr[0]);
	blas_axpby(1, Mr, 1, r, N);
	blas_axpby(1, B, -1, r, N);

	int iterate = 0;
	int iter_max = 1000;
	double tol = 1e-6;
	double tol2 = tol * tol;
	double error2 = blas_dot(r, r,N);
	printf("error2=%f\n", error2);
	while (error2>tol2 && iterate<iter_max) {
		// compute Ar
		matrix_vector_product(S, r, Ar);
		matrix_vector_product(M, r, Mr);
		blas_axpby(1, Mr, 1, Ar, N);

		double alpha = error2 / blas_dot(Ar, r, N);

		// update u
		blas_axpby(alpha, r, 1, U, N);

		//update r
		// Note r_{k+1}=r_k-alpha_k Ar_k
		blas_axpby(-alpha, Ar, 1, r, N);

		//update error
		error2 = blas_dot(r, r, N);
		iterate += 1;
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
void build_cube_mesh(struct Mesh* m, int N);
void send_cube_to_sphere(struct Vertex* vert, int vtx_count);

/******************************************************************************
 * Main routine
 *****************************************************************************/
int main(int argc, char** argv)
{
	if (argc < 2)
		return EXIT_FAILURE;

	struct Mesh m;
	build_cube_mesh(&m, atoi(argv[1]));
	//send_cube_to_sphere(m.vertices, m.vtx_count);
	int N = m.vtx_count;
	int n = m.tri_count;
	printf("Number of DOF : %d\n", N);
	printf("vertices\n");
	for (int i = 0; i < N; i++) {
		printf(" %f", m.vertices[i].x);
		printf(" %f",m.vertices[i].y);
		printf(" %f\n",m.vertices[i].z);
	}
	printf("triangles\n");
	for (int i = 0; i < n; i++) {
		printf(" %d", m.triangles[i].a);
		printf(" %d", m.triangles[i].b);
		printf(" %d\n", m.triangles[i].c);
	}


	struct SparseMatrix M;
	struct SparseMatrix S;
	build_fem_matrices(&m, &S, &M);

	/* Fill F */
	double* F = array(N);
	for (int i = 0; i < N; i++) {
		struct Vertex v = m.vertices[i];
		F[i] = f(v.x, v.y, v.z);
	}
	/* Fill B = MF */
	double* B = array(N);
	matrix_vector_product(&M, F, B);

	/* Solve (S + M)U = B */
	double* U = array(N);
	double error;
	int iter = gradient_system_solve(&S, &M, B, U,&error, N);
	printf("System solved in %d iterations.\n", iter);
	printf("norn of residue(l^2 :%f\n", error);
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
int build_cube_vertices(struct Vertex* vert, int N) {
	int V = N + 1;
	for (int row=0; row < V; row++) {
		for (int col=0; col < V; col++) {
			vert[V * row + col].x = col;
			vert[V * row + col].y = 0.0;
			vert[V * row + col].z = row;

		}
	}
	for (int row = 0; row < V; row++) {
		for (int col = 0; col < V; col++) {
			vert[V * V + V * row + col].x = N;
			vert[V * V+ V* row + col].y = col;
			vert[V * V+ V * row + col].z = row;
		}
	}
	for (int row = 0; row < V; row++) {
		for (int col = 0; col < V; col++) {
			vert[2 * V * V + V * row + col].x = N-col;
			vert[2 * V* V + V * row + col].y = N;
			vert[2 * V * V + V * row + col].z = row;
		}
	}
	for (int row = 0; row < V; row++) {
		for (int col = 0; col < V; col++) {
			vert[3 * V * V + V * row + col].x = 0.0;
			vert[3 * V * V + V * row + col].y = N-col;
			vert[3 * V * V + V * row + col].z = row;
		}
	}
	for (int row = 0; row < V; row++) {
		for (int col = 0; col < V; col++) {
			vert[4 * V * V + V * row + col].x = col;
			vert[4 * V * V + V * row + col].y = N-row;
			vert[4 * V * V + V * row + col].z = 0.0;
		}
	}
	for (int row = 0; row < V; row++) {
		for (int col = 0; col < V; col++) {
			vert[5 * V * V+ V * row + col].x = col;
			vert[5 * V * V + V * row + col].y = row;
			vert[5 * V * V + V * row + col].z = N;
		}
	}
	for (int i = 0; i < 6 * V * V; i++) {
		vert[i].x -= N / 2;
		vert[i].y -= N / 2;
		vert[i].z -= N / 2;
	}
	
	return 6 * V * V;
}

int build_cube_triangles(struct Triangle* tri, int N) {
	int V = N + 1;
	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++) {
			tri[2 * (N * row + col)].a = V * row + col;
			tri[2 * (N * row + col)].b = V * row + col + 1;
			tri[2 * (N * row + col)].c = V * row + col + V + 1;
			tri[2 * (N * row + col) + 1].a = V * row + col;
			tri[2 * (N * row + col) + 1].b = V * row + col + V + 1;
			tri[2 * (N * row + col) + 1].c = V * row + col + V;

		}
	}
	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++) {
			tri[2 * N * N + 2 * (N * row + col)].a = V * V + V * row + col;
			tri[2 * N * N + 2 * (N * row + col)].b = V * V + V * row + col + 1;
			tri[2 * N * N + 2 * (N * row + col)].c = V * V + V * row + col + V + 1;
			tri[2 * N * N + 2 * (N * row + col) + 1].a = V * V + V * row + col;
			tri[2 * N * N + 2 * (N * row + col) + 1].b = V * V + V * row + col + V + 1;
			tri[2 * N * N + 2 * (N * row + col) + 1].c = V * V + V * row + col + V;

		}
	}
	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++) {
			tri[4 * N * N + 2 * (N * row + col)].a = 2 * V * V + V * row + col;
			tri[4 * N * N + 2 * (N * row + col)].b = 2 * V * V + V * row + col + 1;
			tri[4 * N * N + 2 * (N * row + col)].c = 2 * V * V + V * row + col + V + 1;
			tri[4 * N * N + 2 * (N * row + col) + 1].a = 2 * V * V + V * row + col;
			tri[4 * N * N + 2 * (N * row + col) + 1].b = 2 * V * V + V * row + col + V + 1;
			tri[4 * N * N + 2 * (N * row + col) + 1].c = 2 * V * V + V * row + col + V;
		}
	}
	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++) {
			tri[6 * N * N + 2 * (N * row + col)].a = 3 * V * V + V * row + col;
			tri[6 * N * N + 2 * (N * row + col)].b = 3 * V * V + V * row + col + 1;
			tri[6 * N * N + 2 * (N * row + col)].c = 3 * V * V + V * row + col + V + 1;
			tri[6 * N * N + 2 * (N * row + col) + 1].a = 3 * V * V + V * row + col;
			tri[6 * N * N + 2 * (N * row + col) + 1].b = 3 * V * V + V * row + col + V + 1;
			tri[6 * N * N + 2 * (N * row + col) + 1].c = 3 * V * V + V * row + col + V;
		}
	}
	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++) {
			tri[8 * N * N + 2 * (N * row + col)].a = 4 * V * V + V * row + col;
			tri[8 * N * N + 2 * (N * row + col)].b = 4 * V * V + V * row + col + 1;
			tri[8 * N * N + 2 * (N * row + col)].c = 4 * V * V + V * row + col + V + 1;
			tri[8 * N * N + 2 * (N * row + col) + 1].a = 4 * V * V + V * row + col;
			tri[8 * N * N + 2 * (N * row + col) + 1].b = 4 * V * V + V * row + col + V + 1;
			tri[8 * N * N + 2 * (N * row + col) + 1].c = 4 * V * V + V * row + col + V;
		}
	}
	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++) {
			tri[10 * N * N + 2 * (N * row + col)].a = 5 * V * V + V * row + col;
			tri[10 * N * N + 2 * (N * row + col)].b = 5 * V * V + V * row + col + 1;
			tri[10 * N * N + 2 * (N * row + col)].c = 5 * V * V + V * row + col + V + 1;
			tri[10 * N * N + 2 * (N * row + col) + 1].a = 5 * V * V + V * row + col;
			tri[10 * N * N + 2 * (N * row + col) + 1].b = 5 * V * V + V * row + col + V + 1;
			tri[10 * N * N + 2 * (N * row + col) + 1].c = 5 * V * V + V * row + col + V;
		}
	}
	return 12 * N * N;
}

int dedup_mesh_vertices(struct Mesh* m) {
	int v_count = 0;
	int Vtx_count = m->vtx_count;
	int* n_ind = (int*)malloc(Vtx_count * sizeof(int));
	for (int i = 0; i < Vtx_count; i++) {
		bool found = false;
		struct Vertex v1 = m->vertices[i];
		for (int j = 0; j < i; j++) {
			struct Vertex v2 = m->vertices[j];
			if (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z) {
				n_ind[i] = n_ind[j];
				found = true;
				break;
			}
		}
		if (!found) {
			n_ind[i] = v_count;
			v_count++;
		}
	}
	for (int i=0; i < v_count; i++) {
		m->vertices[n_ind[i]] = m->vertices[i];
	}
	for (int i = 0; i < m->tri_count; i++) {
		m->triangles[i].a = n_ind[m->triangles[i].a];
		m->triangles[i].b = n_ind[m->triangles[i].b];
		m->triangles[i].c = n_ind[m->triangles[i].c];

	}
	free(n_ind);
	return v_count;

}

void build_cube_mesh(struct Mesh* m, int N)
{
	/* Number of vertices per side = number of divisions + 1 */
	int V = N + 1;

	/* We allocate for 6 * V^2 vertices */
	int max_vert = 6 * V * V;
	m->vertices = (struct Vertex*)malloc(max_vert * sizeof(struct Vertex));
	m->vtx_count = 0;

	/* We allocate for 12 * N^2 triangles */
	int tri_count = 12 * N * N;
	m->triangles =
		(struct Triangle*)malloc(tri_count * sizeof(struct Triangle));
	m->tri_count = 0;

	/* We fill the vertices and then the faces */
	m->vtx_count = build_cube_vertices(m->vertices, N);
	m->tri_count = build_cube_triangles(m->triangles, N);

	/* We fix-up vertex duplication */
	m->vtx_count = dedup_mesh_vertices(m);
	assert(m->vtx_count == 6 * V * V - 12 * V + 8);

	/* Rescale to unit cube centered at the origin */
	for (int i = 0; i < m->vtx_count; ++i) {
		struct Vertex* v = &m->vertices[i];
		v->x = 2 * v->x / N - 1;
		v->y = 2 * v->y / N - 1;
		v->z = 2 * v->z / N - 1;
	}
}

/******************************************************************************
 * The so-called spherical cube, built by simply normalizing all vertices of
 * the cube mesh so that they end up in S^2
 *****************************************************************************/
void send_cube_to_sphere(struct Vertex* vert, int vtx_count) {
	for (int i = 0; i < vtx_count; i++) {
		double norm = sqrt(vert[i].x * vert[i].x + vert[i].y * vert[i].y + vert[i].z * vert[i].z);
		vert[i].x /= norm;
		vert[i].y /= norm;
		vert[i].z /= norm;
	}
}
/*****************************************************************************/
