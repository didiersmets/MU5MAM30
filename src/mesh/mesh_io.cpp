#ifdef DEBUG
#include <stdio.h>
#endif
#include <string.h>

#define FAST_OBJ_IMPLEMENTATION 1
#include "fast_obj/fast_obj.h"
#undef FAST_OBJ_IMPLEMENTATION

#include "array.h"
#include "hash.h"
#include "hash_table.h"
#include "mesh.h"
#include "mesh_io.h"
#include "vec2.h"
#include "vec3.h"

struct ObjVertexHasher {
	bool has_normals;
	bool has_uv;
	const Vec3 *pos;
	const Vec3 *nml;
	const Vec2 *uv;
	static constexpr fastObjIndex empty_key = { 0, 0, 0 };
	size_t hash(fastObjIndex key) const;
	bool is_empty(fastObjIndex key) const;
	bool is_equal(fastObjIndex key1, fastObjIndex key2) const;
};
typedef HashTable<fastObjIndex, uint32_t, ObjVertexHasher> ObjVertexTable;

inline size_t ObjVertexHasher::hash(fastObjIndex key) const
{
	const uint32_t *p = reinterpret_cast<const uint32_t *>(pos + key.p);
	uint32_t hash = 0;
	hash = murmur2_32(hash, p[0]);
	hash = murmur2_32(hash, p[1]);
	hash = murmur2_32(hash, p[2]);
	return hash;
}

bool ObjVertexHasher::is_empty(fastObjIndex key) const
{
	return (key.p == 0);
}

bool ObjVertexHasher::is_equal(fastObjIndex key1, fastObjIndex key2) const
{
	bool res = (pos[key1.p] == pos[key2.p]);
	if (res && has_normals)
		res &= (nml[key1.n] == nml[key2.n]);
	if (res && has_uv)
		res &= (uv[key1.t] == uv[key2.t]);

	return (res);
}

static int load_obj(const fastObjMesh &obj, Mesh &mesh)
{
	// size_t obj_vertex_count = 0;
	size_t index_count = 0;

	for (unsigned int i = 0; i < obj.face_count; ++i) {
		//	obj_vertex_count += obj.face_vertices[i];
		index_count += 3 * (obj.face_vertices[i] - 2);
	}

	mesh.indices.resize(index_count);

	bool has_normals = false;
	bool has_uv = false;
	// for (size_t i = 0; i < obj_vertex_count; ++i) {
	//	has_normals |= (obj.indices[i].n != 0);
	//	has_uv |= (obj.indices[i].t != 0);
	// }
	// data.vtx_attr |= has_normals ? VtxAttr::NML : 0;
	// data.vtx_attr |= has_uv ? VtxAttr::UV0 : 0;

	size_t vertex_count_guess = index_count / 6;
	vertex_count_guess += vertex_count_guess / 2;
	mesh.positions.reserve(vertex_count_guess);
	mesh.positions.resize(0);

	/* Discover vertices and encode indices */

	size_t idx = 0;
	size_t idx_offset = 0;

	size_t vertex_count = 0;
	ObjVertexHasher hasher{ has_normals, has_uv,
				(const Vec3 *)obj.positions,
				(const Vec3 *)obj.normals,
				(const Vec2 *)obj.texcoords };
	ObjVertexTable vertices(vertex_count_guess, hasher);

	for (size_t i = 0; i < obj.face_count; ++i) {
		size_t idx_start = idx;
		for (size_t j = 0; j < obj.face_vertices[i]; ++j) {
			if (j >= 3) /* Triangulate */
			{
				mesh.indices[idx + 0] = mesh.indices[idx_start];
				mesh.indices[idx + 1] = mesh.indices[idx - 1];
				idx += 2;
			}

			fastObjIndex pnt = obj.indices[idx_offset + j];

			uint32_t *p = vertices.get_or_set(pnt, vertex_count);
			if (!p) {
				mesh.indices[idx] = vertex_count;

				/* Copy vertex */
				{
					float *v = obj.positions + 3 * pnt.p;
					Vec3 new_vertex(v[0], v[1], v[2]);
					mesh.positions.push_back(new_vertex);
				}

				vertex_count++;
			} else {
				mesh.indices[idx] = *p;
			}
			idx++;
		}
		idx_offset += obj.face_vertices[i];
	}
	assert(vertex_count == mesh.positions.size);

	// bool shrink = true;
	// mesh.positions.resize(vertex_count, shrink);

	return (EXIT_SUCCESS);
}

int load_obj(const char *filename, Mesh &mesh)
{
	fastObjMesh *obj = fast_obj_read(filename);
	if (obj == nullptr) {
		return (EXIT_FAILURE);
	}
	int res = load_obj(*obj, mesh);
	fast_obj_destroy(obj);

	return (res);
}
