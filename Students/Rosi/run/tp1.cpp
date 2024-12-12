#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hash_table.h"

#define FAST 0
#define CONJ 0
#define CRSMATRIX 0


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

struct CRSMatrix {
	int rows;
	int cols;
	int nnz;
	int *row_ptr;
	int *col_ind;
	double *val;
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

void matrix_vector_product(const struct CRSMatrix *M, const double *v,
			   double *Mv){
	for(int i = 0; i < M->rows; i++) Mv[i] = 0;

	for (int i = 1; i < M->rows; i++) {
		for (int k = M->row_ptr[i - 1]; k < M->row_ptr[i]; k++) {
			Mv[i - 1] += M->val[k] * v[M->col_ind[k]];
		}
	}
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
        
        struct Coeff * mass = &M->coeffs[9 * t];

        mass[0] = (Coeff) {a, a, area/6};
        mass[1] = (Coeff) {b, b, area/6};
        mass[2] = (Coeff) {c, c, area/6};
        mass[3] = (Coeff) {a, b, area/12};
        mass[4] = (Coeff) {a, c, area/12};
        mass[5] = (Coeff) {b, a, area/12};
        mass[6] = (Coeff) {c, a, area/12};
        mass[7] = (Coeff) {c, b, area/12};
        mass[8] = (Coeff) {b, c, area/12};

        struct Coeff * stiff = &S->coeffs[9*t];
        double mult = 1. / (4*area);

        stiff[0] = (Coeff) {a,a,dot(BC,BC) * mult};
        stiff[1] = (Coeff) {b,b,dot(CA,CA) * mult};
        stiff[2] = (Coeff) {c,c,dot(AB,AB) * mult};
		stiff[3] = (Coeff) {a,b,dot(CA,BC) * mult};
        stiff[4] = (Coeff) {b,a,dot(CA,BC) * mult};
        stiff[5] = (Coeff) {a,c,dot(AB,BC) * mult};
        stiff[6] = (Coeff) {c,a,dot(AB,BC) * mult};
        stiff[7] = (Coeff) {b,c,dot(AB,CA) * mult};
        stiff[8] = (Coeff) {c,b,dot(AB,CA) * mult};
	
	}
	return;
}

void build_fem_matrices(const struct Mesh *m, struct CRSMatrix *S,
			struct CRSMatrix *M){
	
	int N = m->vtx_count;
	S->cols = S->rows = M->cols = M->rows = N;

	S->nnz = 9*m->tri_count;

	S->row_ptr = (int *) malloc((N + 1) * sizeof(int));
	M->row_ptr = (int *) malloc((N + 1) * sizeof(int));

	for (int i = 0; i < N + 1; i++) {
		S->row_ptr[i] = 1;
		M->row_ptr[i] = 1;
	}

	for(int t = 0; t < m->tri_count; t++){
		S->row_ptr[m->triangles[t].a]++;
		S->row_ptr[m->triangles[t].b]++;
		S->row_ptr[m->triangles[t].c]++;
		M->row_ptr[m->triangles[t].a]++;
		M->row_ptr[m->triangles[t].b]++;
		M->row_ptr[m->triangles[t].c]++;
	}

	for(int i = 0; i < N; i++){
		S->row_ptr[i + 1] += S->row_ptr[i];
		M->row_ptr[i + 1] += M->row_ptr[i];
	}


	for(int i = N-1; i >= 0; i--){
		S->row_ptr[i + 1] = S->row_ptr[i];
		M->row_ptr[i + 1] = M->row_ptr[i];
	}

	S->row_ptr[0] = 0;
	M->row_ptr[0] = 0;

	S->nnz = S->row_ptr[N];
	M->nnz = M->row_ptr[N];

	//filling the matrices

	S->val = (double *) malloc(S->nnz * sizeof(double));
	S->col_ind = (int *) malloc(S->nnz * sizeof(int));
	M->val = (double *) malloc(M->nnz * sizeof(double));
	M->col_ind = (int *) malloc(M->nnz * sizeof(int));

	for(int i = 0; i < S->nnz; i++){
		S->col_ind[i] = -1;
		M->col_ind[i] = -1;
	}
	
	

	for(int t = 0; t < m->tri_count; t++){

		int a = m->triangles[t].a;
		int b = m->triangles[t].b;
		int c = m->triangles[t].c;
		int tria[3] = {a,b,c};

		for (int k = 0; k < 3; k++) {
            int current_vtx = tria[k];
            for (int j = 0; j < 3; j++) {
                int current_row = tria[j];
                int offset = S->row_ptr[current_row];
                int length = S->row_ptr[current_row + 1] - offset;

                for (int i = 0; i < length; i++) {
                    if (S->col_ind[offset + i] == current_vtx) {
                        break; // Already appeared, skip
                    } else if (S->col_ind[offset + i] == -1) { // First appearance, insert
                        S->col_ind[offset + i] = current_vtx;
                        break;
                    }
                }
			}
		}

		for (int row = 0; row < S->rows; row++) {
    	    int offset = S->row_ptr[row];
    	    int length = S->row_ptr[row + 1] - offset;
    	    qsort(&S->col_ind[offset], length, sizeof(int), [](const void *a, const void *b) {
		        return *(int *)a - *(int *)b;
		    });
    	}

	}

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

		double mult = 1. / (4*area);

		int tria[3] = {a,b,c};

		for(int i = 0; i < 3; i++){
			int current_row = tria[i];
			int offset = M->row_ptr[current_row];
			int length = M->row_ptr[current_row + 1] - offset;
			for(int k = 0; k < length; k++){
				int col = M->col_ind[offset + k];
				int j = -1;
				if(col == a)
					j = 0;
				else if(col == b)
					j = 1;
				else if(col == c)
					j = 2;
				else
					j = -1;
				if(j != -1){
					M->val[offset + k] += area/12 * (j == i ? 2 : 1);
					if(i == 0 && j == 0)
						S->val[offset + k] += dot(BC,BC) * mult;
					else if(i == 1 && j == 1)
						S->val[offset + k] += dot(CA,CA) * mult;
					else if(i == 2 && j == 2)
						S->val[offset + k] += dot(AB,AB) * mult;
					else if(i == 0 && j == 1)
						S->val[offset + k] += dot(CA,BC) * mult;
					else if(i == 0 && j == 2)
						S->val[offset + k] += dot(AB,BC) * mult;
					else if(i == 1 && j == 0)
						S->val[offset + k] += dot(CA,BC) * mult;
					else if(i == 1 && j == 2)
						S->val[offset + k] += dot(AB,CA) * mult;
					else if(i == 2 && j == 0)
						S->val[offset + k] += dot(AB,BC) * mult;
					else if(i == 2 && j == 1)
						S->val[offset + k] += dot(AB,CA) * mult;
				}
	
			}
		}
	}
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
	
	memset(U, 0, N * sizeof(double));

    //initial residue

    double *r = array(N);

	memcpy(r, B, N * sizeof(double));
	double error2 = blas_dot(r,r,N);

    double *Mr = array(N);
    double *Ar = array(N);

    
    int iterate = 0;
    //int iter_max = 1000;
    double tol = 1e-6;
    double tol2 = tol*tol;
    

	#if CONJ == 1
		double *p = r;
		double *Mp = array(N);
		double *Ap = array(N);

	#endif

    while(error2 > tol2){

		#if CONJ == 0

			matrix_vector_product(S,r,Ar);
        	matrix_vector_product(M,r,Mr);
        	blas_axpby(1,Mr,1,Ar,N);



			double alpha = error2/blas_dot(Ar,r,N);
	
        	//Update U
        	blas_axpby(alpha,r,1,U,N);
    
  	    	//Update r (note r{k+1} = r{k} - aplha*(Ar{k}))
    	    blas_axpby(-alpha,Ar,1,r,N);

		#else

			matrix_vector_product(M,p,Mp);
			matrix_vector_product(S,p,Ap);
			blas_axpby(1,Mp,1,Ap,N);

			double pAp = blas_dot(p,Ap,N);
			
			double alpha = blas_dot(p,r,N)/pAp;

			blas_axpby(alpha,p,1,U,N);

			blas_axpby(-alpha,Ap,1,r,N);

			matrix_vector_product(S,r,Ar);
        	matrix_vector_product(M,r,Mr);
        	blas_axpby(1,Mr,1,Ar,N);

			double beta = blas_dot(p,Ar,N)/pAp;

			blas_axpby(1,r,-beta,p,N);
			
		#endif
    	//Update error
        error2 = blas_dot(r,r,N);

        iterate++;
    }
    free(Mr);
    free(Ar);
    free(r);

	#if CONJ == 1
		free(Mp);
		free(Ap);
		free(p);
	#endif

    *error = sqrt(error2);
    
    return iterate;

}

int gradient_system_solve(const struct CRSMatrix *S,
			  const struct CRSMatrix *M, const double *B,
			  double *U,double *error, int N){
	
	memset(U, 0, N * sizeof(double));

	//initial residue

	double *r = array(N);

	memcpy(r, B, N * sizeof(double));
	double error2 = blas_dot(r,r,N);

	double *Mr = array(N);
	double *Ar = array(N);

	
	int iterate = 0;
	//int iter_max = 1000;
	double tol = 1e-6;
	double tol2 = tol*tol;
	

	#if CONJ == 1
		double *p = r;
		double *Mp = array(N);
		double *Ap = array(N);

	#endif

	while(error2 > tol2){

		

		#if CONJ == 0

			matrix_vector_product(S,r,Ar);
			matrix_vector_product(M,r,Mr);
			blas_axpby(1,Mr,1,Ar,N);

			double alpha = error2/blas_dot(Ar,r,N);
	
        	//Update U
        	blas_axpby(alpha,r,1,U,N);
    
  	    	//Update r (note r{k+1} = r{k} - aplha*(Ar{k}))
    	    blas_axpby(-alpha,Ar,1,r,N);

		#else

			matrix_vector_product(M,p,Mp);
			matrix_vector_product(S,p,Ap);
			blas_axpby(1,Mp,1,Ap,N);

			double pAp = blas_dot(p,Ap,N);
			
			double alpha = blas_dot(p,r,N)/pAp;

			blas_axpby(alpha,p,1,U,N);

			blas_axpby(-alpha,Ap,1,r,N);

			matrix_vector_product(S,r,Ar);
        	matrix_vector_product(M,r,Mr);
        	blas_axpby(1,Mr,1,Ar,N);

			double beta = blas_dot(p,Ar,N)/pAp;

			blas_axpby(1,r,-beta,p,N);
			
		#endif
    	//Update error
        error2 = blas_dot(r,r,N);

        iterate++;
    }
    free(Mr);
    free(Ar);
    free(r);

	#if CONJ == 1
		free(Mp);
		free(Ap);
		free(p);
	#endif

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

		struct SparseMatrix M1;
		struct SparseMatrix S1;
		build_fem_matrices(&m, &S1, &M1);

		//print of S1 matrix
		for(int i = 0; i < S1.nnz; i++){
			printf("S1[%d][%d] = %f\n", S1.coeffs[i].i, S1.coeffs[i].j, S1.coeffs[i].val);
		}
	
 
	
		struct CRSMatrix M2;
		struct CRSMatrix S2;
		build_fem_matrices(&m, &S2, &M2);

		//print of S2 matrix
		for(int i = 1; i < S2.rows;i++){
			for(int j = S2.row_ptr[i-1]; j < S2.row_ptr[i]; j++){
				printf("S2[%d][%d] = %f\n", i-1, S2.col_ind[j], S2.val[j]);
			}
		}
		
	
	double *U1 = array(N);
	double *U2 = array(N);

	/* Fill F */
	double *F = array(N);
	for (int i = 0; i < N; i++) {
		struct Vertex v = m.vertices[i];
		F[i] = f(v.x, v.y, v.z);
	}

	matrix_vector_product(&S1, F, U1);
	matrix_vector_product(&S2, F, U2);

	blas_axpby(1, U1, -1, U2, N);
	for (int i = 0; i < N; i++)
		assert(fabs(U2[i]) < 1e-10);

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
    int V = N + 1;
	assert(V > 0);
	int NVF = V * V; // Number of vertices per face
	//double mult = 2. / (V - 1); //Better not use this as by floating points errors
	int k = 0;

	for (int i = 0; i < V; i++) {
		for (int j = 0; j < V; j++) {

			vert[0*NVF + k].x = j;
			vert[0*NVF + k].y = 0;
			vert[0*NVF + k].z = i;

			vert[1*NVF + k].x = N;
			vert[1*NVF + k].y = j;
			vert[1*NVF + k].z = i;

			vert[2*NVF + k].x = N - j;
			vert[2*NVF + k].y = N;
			vert[2*NVF + k].z = i;

			vert[3*NVF + k].x = 0;
			vert[3*NVF + k].y = N - j;
			vert[3*NVF + k].z = i;

			vert[4*NVF + k].x = j;
			vert[4*NVF + k].y = N - i;
			vert[4*NVF + k].z = 0;

			vert[5*NVF + k].x = j;
			vert[5*NVF + k].y = i;
			vert[5*NVF + k].z = N;

			k++; // in the end it will be equal to NVF - 1
		}
	}
	return 6 * NVF;
}

int build_cube_triangles(struct Triangle *tri, int N){
    int V = N + 1;
	int t = 0;
	for (int face = 0; face < 6; face++) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				int v = face * V * V + i * V + j;
				tri[t++] = {v, v + 1, v + 1 + V};
				tri[t++] = {v, v + 1 + V, v + V};
			}
		}
	}
	assert(t == 12 * N * N);
	return t;
}


#if FAST == 1
int dedup_mesh_vertices(struct Mesh *m){

	HashTable<int, int> remap(m->vtx_count);
	int Ntot = m->vtx_count;
	int NVF = Ntot / 6; 
	int N = (int) sqrt(NVF + 0.5); //0.5 just not to have problems with floating points
	assert(N * N == NVF);
	int vtx_count = 0;


	for(int i = 0; i < Ntot; i++){
		
		if(remap.get(m->vertices[i].x + (N + 1) * m->vertices[i].y + (N + 1)*(N + 1)*m->vertices[i].z) == nullptr){
			remap.set_at(m->vertices[i].x + (N + 1) * m->vertices[i].y + (N + 1)*(N + 1)*m->vertices[i].z , vtx_count);
			vtx_count++;
		}
		/*else{ //not necessary
			remap.set_at(m->vertices[i].x + (N + 1) * m->vertices[i].y + (N + 1)*(N + 1)*m->vertices[i].z , *remap.get(m->vertices[i].x + (N + 1) * m->vertices[i].y + (N + 1)*(N + 1)*m->vertices[i].z));
		}*/
	}
	
	for(int i = 0; i < m->tri_count; i++){

		struct Triangle *T = &m->triangles[i];

		T->a = *remap.get(m->vertices[T->a].x + (N + 1) * m->vertices[T->a].y + (N + 1)*(N + 1)*m->vertices[T->a].z);
		assert(T->a < vtx_count);
		T->b = *remap.get(m->vertices[T->b].x + (N + 1) * m->vertices[T->b].y + (N + 1)*(N + 1)*m->vertices[T->b].z);
		assert(T->b < vtx_count);
		T->c = *remap.get(m->vertices[T->c].x + (N + 1) * m->vertices[T->c].y + (N + 1)*(N + 1)*m->vertices[T->c].z);
		assert(T->c < vtx_count);
	}

	for(int i = 0; i < m->vtx_count; i++){
		int *v = remap.get(m->vertices[i].x + (N + 1) * m->vertices[i].y + (N + 1)*(N + 1)*m->vertices[i].z);
		m->vertices[*v] = m->vertices[i];
	}
	
	return vtx_count;
}

#else

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
#endif

void build_cube_mesh(struct Mesh *m, int N)
{
	// Number of vertices per side = number of divisions + 1 
	int V = N + 1;

	// We allocate for 6 * V^2 vertices 
	int max_vert = 6 * V * V;
	m->vertices = (struct Vertex *)malloc(max_vert * sizeof(struct Vertex));
	m->vtx_count = 0;

	// We allocate for 12 * N^2 triangles 
	int tri_count = 12 * N * N;
	m->triangles = (struct Triangle *)malloc(tri_count * sizeof(struct Triangle));
	m->tri_count = 0;

	// We fill the vertices and then the faces 
	m->vtx_count = build_cube_vertices(m->vertices, N);
	m->tri_count = build_cube_triangles(m->triangles, N);

	// We fix-up vertex duplication 
	m->vtx_count = dedup_mesh_vertices(m);
	assert(m->vtx_count == 6 * V * V - 12 * V + 8);

	// Rescale to unit cube centered at the origin 

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
    for (int i = 0; i < vtx_count; i++) {
		struct Vertex *v = &vert[i];
		double norm = sqrt(v->x * v->x + v->y * v->y + v->z * v->z);
		v->x /= norm;
		v->y /= norm;
		v->z /= norm;
	}
}
/*****************************************************************************/