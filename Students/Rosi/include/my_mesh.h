#pragma once

#include <iostream>
#include <string.h>
#include "vec3.h"


struct Mesh {
	int vtx_count;
	int tri_count;
	struct TVec3<double> *vertices;
	struct TVec3<int> *triangles;
};


int build_cube_vertices(struct TVec3<double> *vert, int N);
int build_cube_triangles(struct TVec3<int> *tri, int N);

int dedup_mesh_vertices(struct Mesh *m);
void build_cube_mesh(struct Mesh *m, int N);
void send_cube_to_sphere(struct TVec3<double> *vert, int vtx_count);