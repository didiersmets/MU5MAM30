#include <math.h>

#include "my_mesh.h"
#include "vec3.h"
#include "hash_table.h"


int build_cube_vertices(struct TVec3<double> *vert, int N){
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

int build_cube_triangles(struct TVec3<int> *tri, int N){
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

		struct TVec3<int> *T = &m->triangles[i];

		T->x = *remap.get(m->vertices[T->x].x + (N + 1) * m->vertices[T->x].y + (N + 1)*(N + 1)*m->vertices[T->x].z);
		assert(T->x < vtx_count);
		T->y = *remap.get(m->vertices[T->y].x + (N + 1) * m->vertices[T->y].y + (N + 1)*(N + 1)*m->vertices[T->y].z);
		assert(T->y < vtx_count);
		T->z = *remap.get(m->vertices[T->z].x + (N + 1) * m->vertices[T->z].y + (N + 1)*(N + 1)*m->vertices[T->z].z);
		assert(T->z < vtx_count);
	}

	for(int i = 0; i < m->vtx_count; i++){
		int *v = remap.get(m->vertices[i].x + (N + 1) * m->vertices[i].y + (N + 1)*(N + 1)*m->vertices[i].z);
		m->vertices[*v] = m->vertices[i];
	}
	
	return vtx_count;
}

void build_cube_mesh(struct Mesh *m, int N)
{
	// Number of vertices per side = number of divisions + 1 
	int V = N + 1;

	// We allocate for 6 * V^2 vertices 
	int max_vert = 6 * V * V;
	m->vertices = (struct TVec3<double> *)malloc(max_vert * sizeof(struct TVec3<double>));
	m->vtx_count = 0;

	// We allocate for 12 * N^2 triangles 
	int tri_count = 12 * N * N;
	m->triangles = (struct TVec3<int> *)malloc(tri_count * sizeof(struct TVec3<int>));
	m->tri_count = 0;

	// We fill the vertices and then the faces 
	m->vtx_count = build_cube_vertices(m->vertices, N);
	m->tri_count = build_cube_triangles(m->triangles, N);

	// We fix-up vertex duplication 
	m->vtx_count = dedup_mesh_vertices(m);
	assert(m->vtx_count == 6 * V * V - 12 * V + 8);

	// Rescale to unit cube centered at the origin 

	for (int i = 0; i < m->vtx_count; ++i) {
		struct TVec3<double> *v = &m->vertices[i];
		v->x = 2 * v->x / N - 1;
		v->y = 2 * v->y / N - 1;
		v->z = 2 * v->z / N - 1;
	}
}


void send_cube_to_sphere(struct TVec3<double> *vert, int vtx_count){
    for (int i = 0; i < vtx_count; i++) {
		struct TVec3<double> *v = &vert[i];
		double norm = sqrt(v->x * v->x + v->y * v->y + v->z * v->z);
		v->x /= norm;
		v->y /= norm;
		v->z /= norm;
	}
}
/*****************************************************************************/