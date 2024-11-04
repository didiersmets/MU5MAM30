//  TP_C++.cpp
//  TP1-EF
//  Created by DJ H on 2024/11/3.

/*C语言预处理指令*/
#include <assert.h> /*查错,常用的函数是 assert()*/
#include <math.h>
#include <stdbool.h>/*引入bool类型函数,使用 true 和 false */
#include <stdio.h>/*允许程序进行输入输出操作，处理文本和数据的读取与写入,如 printf()、scanf()*/
#include <stdlib.h>
/*通用工具函数，包括动态内存分配（malloc()、calloc()、free()）、随机数生成（rand()、srand()）、程序退出（exit()）等*/
#include <string.h> /*处理字符串和内存的函数，如 strlen()、strcpy()、strcat()、strcmp() 等。*/

/* 此TP是为了熟悉C语言,并求解方程
 *    - \Delta u + u = f
 *
 * on (e.g.) the sphere S^2. The sphere is handy for testing because
 * "we" know explicit solutions (besides constants) for some particular
 * cases of f (which ones ?), but process would work on an arbitrary surface
 * mesh.
 *
 * provide some conceptual (algorithmmic/performance/robustness) improvements
 * 自己做一些概念上的（算法/性能/鲁棒性）改进 */


/* The C structures we define for our needs 为了满足特定需求而定义的C语言结构 */

/* A Vertex is simply a point in R^3.
 * Vertex结构体是三维空间的一个点
 * double表示数据类型 */
struct Vertex {
    double x;
    double y;
    double z;
};

/* A triangle refers to three vertices by their index
 * Triangle结构体是三角形,通过三个顶点的索引表示 */
struct Triangle {
    int a;
    int b;
    int c;
};

/* A mesh is an array of vertices, and an array of triangles
 * built over these vertices
 * mesh结构体是网络,由顶点数组和三角形数组构成
 * vtx_count 表示顶点数量
 * tri_count 表示三角形数量
 * *vertices指向Vertex类型数据的指针,用于存储所有顶点
 * *triangles指向Triangle类型数据的指针,用于存储所有三角形
 */
struct Mesh {
    int vtx_count;
    int tri_count;
    struct Vertex *vertices;
    struct Triangle *triangles;
};

/* A coefficient of a sparse matrix
 * Coeff结构体表示稀疏矩阵的一个系数
 *包含i、j两个成员表示行列索引,val表示对因位置的值
 */
struct Coeff {
    int i;
    int j;
    double val;
};

/* A Sparse matrix is an (order independent) array of coeffs,
 * the non zero entries of the matrix.
 * 此结构体表示一个稀疏矩阵,由系数组成,包含所有非零元素
 * rows表示矩阵的行数;
 * nnz表示非零元素的数量;
 * coeffs: 指向Coeff类型数组的结构体,用来存储非零系数
 * 是最简单的,但不是最高效的
 */
struct SparseMatrix {
    int rows;
    int cols;
    int nnz;
    struct Coeff *coeffs;
};

/* A Vector in R^3. Similar to Vertex, but as in affine geometry we may
 * wish to distinguish vectors and points.
 * Vector结构体表示三维空间中的向量,和Vertex类似,但在仿射几何中需要区分向量和点
 * x表示向量在x轴的分量
 */
struct Vector {
    double x;
    double y;
    double z;
};

/******************************************************************************
 * Vectors in R^3 定义三维空间中向量的操作,例如向量的创建、点积、范数、叉积
 *********************************************************************************************/

/* A vector from its end points
 * 从A和B点创建向量
 * 定义一个名为vector的函数,接受两个Vertex类型的参数A和B, 返回一个Vector类型的结果 */
struct Vector vector(struct Vertex A, struct Vertex B)
{
    struct Vector res;
    res.x = B.x - A.x;
    res.y = B.y - A.y;
    res.z = B.z - A.z;
    return res;
}

/*点积函数:定义一个叫dot的函数,接受两个Vector类型的参数V和W,返回他们的点积(数据类型是double) */
double dot(struct Vector V, struct Vector W)
{
    return (V.x * W.x + V.y * W.y + V.z * W.z);
}

/*范数函数*/
double norm(struct Vector V)
{
    return sqrt(dot(V, V));}

/*叉积函数,250那门课学的*/
struct Vector cross(struct Vector V, struct Vector W)
{
    struct Vector res;
    res.x = V.y * W.z - V.z * W.y;
    res.y = V.z * W.x - V.x * W.z;
    res.z = V.x * W.y - V.y * W.x;
    return res;
}


/******************************************************************************
 * Computes the product M * v when M is a SparseMatrix 计算矩阵-向量乘积
 *****************************************************************************/
/*void表示函数没有返回值;如果在参数列表使用表示该函数没有参数
 *参数:struct SparseMatrix *M 表示M是SparseMatrix类型的数据
 *参数:const表示M是只读的,函数不会修改M所指向的系数矩阵内容
 *参数:double *Mv表示Mv是double数据类型的乘积结果,结果会在函数中被修改
 *先初始化Mv,初始值是0: M->rows表示调用M系数矩阵的行数这个参数; r++表示r=r+1
 *再遍历M的coeffs每个非零元素: M->coeffs[k]表示第k个非零项目; i表示这个非零项对应的行
 *assert()用于确保索引i在矩阵的有效范围内,否则程序会停止
 */
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
 * Builds the P1 stiffness and mass matrices of a given mesh.构建刚度矩阵S和质量矩阵M的函数
 * We do not try to assemble different elements together here for simplicity.
 * Both matrices M and S will therefore have 9 * number of triangles.
 * We derived the formulas in the first lecture.
 *****************************************************************************/
void build_fem_matrices(const struct Mesh *m, struct SparseMatrix *S,
            struct SparseMatrix *M)
{
    int N = m->vtx_count; //获取网格的顶点数量
    S->rows = S->cols = M->rows = M->cols = m->vtx_count;  //设置S和M的行列数为顶点数量
    /* We do not try to assemble, so 9 coeffs per triangle */
    S->nnz = M->nnz = 9 * m->tri_count; // 设置非零项数量为三角形数量的9倍
    /*这行代码为S->coeffs动态分配了存储S->nnz个struct Coefff非零项的内存区间
     *通过这个数组,矩阵S可以存储他所有的非零系数,构建出系数矩阵所需要的数据
     *(struct Coeff *)  malloc返回的是void*类型的指针,所以需要将其强制转换为合适的类型
     *malloc是内存分配函数,动态分配一段指定大小的内存
     *sizeof(struct Coeff)返回struct Coeff的字节大小
     */
    S->coeffs = (struct Coeff *)malloc(S->nnz * sizeof(struct Coeff));
    M->coeffs = (struct Coeff *)malloc(M->nnz * sizeof(struct Coeff));

    //遍历所有三角形,计算每个三角形对应的系数
    for (int i = 0; i < m->tri_count; i++) {
        int a = m->triangles[i].a; //获取第i个三角形的顶点索引a,b,c
        int b = m->triangles[i].b;
        int c = m->triangles[i].c;
        assert(a < N);
        assert(b < N);
        assert(c < N);
        //获取顶点a、b、c的坐标
        struct Vertex A = m->vertices[a];
        struct Vertex B = m->vertices[b];
        struct Vertex C = m->vertices[c];
        //计算向量AB、BC、CA
        struct Vector AB = vector(A, B);
        struct Vector BC = vector(B, C);
        struct Vector CA = vector(C, A);
        //计算三角形面积:叉积的范数等于这俩向量所构成的平行四边形的面积,三角形面积= 1/2 * |a||b|sinC
        struct Vector CAxAB = cross(CA, AB);
        double area = 0.5 * norm(CAxAB);
        
        /*质量矩阵M的系数
         *C++的写法    m[0] = {a, a, area / 6};
         *C的写法:逐个赋值
         *       m[0].i = a;
         *       m[0].j = a;
         *       m[0].val = area / 6;
         *       如果符合C99标准,也可以写mass[1] = (struct Coeff){b, b, area / 6};
         *或者定义一个辅助函数来简化赋值
         *void set_coeff(struct Coeff *coeff, int i, int j, double val) {
         *coeff->i = i;
         *coeff->j = j;
         *coeff->val = val;}
         *然后实际参数赋值中使用set_coeff(&m[1], b, b, area / 6);
         
         *每个三角形三个顶点,质量刚度矩阵是顶点间相互作用,所以得到3*3矩阵
         *&是取地址符号,获取M->coeffs数组中某个未知的内存地址,然后赋值给m
         */
        struct Coeff *m = &M->coeffs[9 * i];
        m[0] = {a, a, area / 6}; //M->coeffs[9] = {a, a, area/6}, 当i=1时
        m[1] = {b, b, area / 6}; //M->coeffs[10] = {a, a, area/6}, 当i=1时
        m[2] = {c, c, area / 6};
        m[3] = {a, b, area / 12};
        m[4] = {b, a, area / 12};
        m[5] = {a, c, area / 12};
        m[6] = {c, a, area / 12};
        m[7] = {b, c, area / 12};
        m[8] = {c, b, area / 12};

        //刚度矩阵
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
 * Routines for elementary linear algebra in arbitrary (large) dimensions  高维初等线性代数(向量)
 *****************************************************************************/

/* Create an (unitialized) array of N double precision floating point values */
/* array函数:创建一个包含N个双精度浮点数值的数组,设置该数组占内存(N*double的内存) */
double *array(int N) {
    return (double *)malloc(N * sizeof(double));
}

/* Vector product between two vectors in dim N
 *blas_dot函数计算N维向量点积
 */
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
 * blas_axpby函数遍历向量的每个分量,计算线性组合Y=ax+bY
 */
void blas_axpby(double a, const double *X, double b, double *Y, int N)
{
    for (int i = 0; i < N; i++) {
        Y[i] = a * X[i] + b * Y[i];
    }
}

/******************************************************************************
 * Solving AU=B where A is SPD of size NxN using steepest descent method.最速下降法求解线性方程AU = B
 * One minimizes the functional 1/2 <AU,U> - <B,U>. 也就是求前面这个函数的最小值
 * The minor peculiarity here is that A = S + M and we do not wish to add these two sparse matrices up-front but simply compute AU as SU + MU wherever needed.
 * 不希望将这两个系数矩阵提前相加,只是希望在需要时分别计算SU和MU来得到AU
 *****************************************************************************/
int gradient_system_solve(const struct SparseMatrix *S,
                          const struct SparseMatrix *M, const double *B,
                          double *U, double *error, int N)
{
    /* Since S + M is symmetric and positive definite we can solve the system by the gradient descent method.
     * This is by far not the best method for ill conditionned matrices, but the point here is not (yet) optimality.
     *
     * We need to solve AU = B with A := (S + M).
     * The iterates are constructed as follows : given an approximation U_k we compute the residue
     * r_k = B - AU_k
     *
     * Then we follow the negative gradient and build(以下公式calcul haute第二节课有)
     *
     *                           <r_k, r_k>
     *          U_{k+1} = U_k +  ----------- r_k =: U_k + alpha_k r_k
     *                           <Ar_k, r_k>
     *
     * until the residue becomes smaller to a given threshold阈值.
     *
     * Note : r_{k + 1} = B - A(U_k + alpha_k r_k) = r_k - alpha_k Ar_k
     *        That will save us one matrix vector product.节省一个矩阵和向量的计算
     */

    /* Set U_0 to the zero vector
     * 初始向量设置为零向量
     * menset函数对内存进行操作,将内存设置为字节级别的值(例如0)
     * U是数组, 每个字节设置为0, U数组中的N和元素占用的总内存大小是N * sizeof(double)(全设置成零) */
    memset(U, 0, N * sizeof(double));

    /* r_0 = B - (M+S)U_0 = B
     * 创建大小为N的数组r来存储残差向量rk
     * mencpy函数将某一块内存中的内容直接复制到另一块内存区域,无视数据类型
     * r是目标内存区域(目标数组), U是源内存区域, N * sizeof(double)表示要复制的字节数 */
    double *r = array(N);
    memcpy(r, B, N * sizeof(double));

    /* l^2 (squared) norm of the residue
     * 计算残差向量r的平方范数 */
    double error2 = blas_dot(r, r, N);

    /* Temporaries
     * 两个大小为N的临时数组 */
    double *Mr = array(N);
    double *Ar = array(N);

    double tol2 = 1e-6; //容差,用来和残差比较,判断算法收敛性
    int iterate = 0;
    while (error2 > tol2) {
        iterate++;
        /* Compute Ar_k */
        matrix_vector_product(S, r, Ar);
        matrix_vector_product(M, r, Mr);
        blas_axpby(1, Mr, 1, Ar, N);    //计算Ar = Mr + Ar
        assert(blas_dot(Ar, r, N) > 0); //验证<Ar, r> > 0,确保系统正定
        /* Compute alpha_k */
        double alpha = blas_dot(r, r, N) / blas_dot(Ar, r, N);

        /* Update U */
        blas_axpby(alpha, r, 1, U, N); // U = U + alpha * r

        /* Update r */
        blas_axpby(-alpha, Ar, 1, r, N); //r = r - alpha * Ar

        /* Update error2 */
        error2 = blas_dot(r, r, N);
    }
    /* Release system memory */
    free(Ar);
    free(Mr);
    free(r);
    error = sqart(error2);
    return iterate, error;
}

/******************************************************************************
 * Let's choose our right hand side f of -\Delta u + u = f 定义右端项函数f
 *****************************************************************************/
double f(double x, double y, double z)
{
    (void)z; /* Avoids compiler warning about unused variable避免编译器警告的技巧,明确表示了z不会被用到*/
    return x * x - y * y;  /* f = xˆ2 - yˆ2*/
}

/******************************************************************************
 * Mesh construction routine declared here, defined later below main routine.就是说一下接下来会定义这两个函数
 *****************************************************************************/
void build_cube_mesh(struct Mesh *m, int N);  //用来构建立方体网格
void send_cube_to_sphere(struct Vertex *vert, int vtx_count); // 将立方体网格变成球面网格

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
