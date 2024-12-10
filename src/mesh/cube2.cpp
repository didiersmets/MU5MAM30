#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#include "cube.h"
#include "mesh.h"
#include "vec3.h"
#include "vec2.h"
#include "hash.h"
#include "hash_table.h"

typedef TVec2<uint32_t> Vec2u;
#define DISSECT_THRESH 4

struct Grid3DHasher {
	static constexpr size_t empty_key = ~static_cast<size_t>(0);
	size_t hash(size_t key) const
	{
		return murmur2_64(0, key);
	}

	bool is_empty(size_t key) const
	{
		return (key == empty_key);
	}
	bool is_equal(size_t key1, size_t key2) const
	{
		return key1 == key2;
	}
};

typedef HashTable<size_t, uint32_t, Grid3DHasher> GridTable;

static void dissect_rect(Vec2u *__restrict vtx, Vec2u vmin, Vec2u vmax)
{
	uint32_t dx = vmax[0] - vmin[0];
	uint32_t dy = vmax[1] - vmin[1];
	if (dx <= DISSECT_THRESH && dy <= DISSECT_THRESH) {
		return;
	}
	bool dir = dx >= dy ? 0 : 1;
	uint32_t sep = (vmax[dir] + vmin[dir]) / 2;

	Vec2u v0min = vmin;
	Vec2u v0max = vmax;
	v0max[dir] = sep - 1;

	Vec2u v1min = vmin;
	Vec2u v1max = vmax;
	v1min[dir] = sep + 1;

	uint32_t v0num = (v0max[0] - v0min[0] + 1) * (v0max[1] - v0min[1] + 1);
	uint32_t v1num = (v1max[0] - v1min[0] + 1) * (v1max[1] - v1min[1] + 1);

	uint32_t v1off = v0num;
	uint32_t vsoff = v0num + v1num;

	uint32_t src = 0;
	while (src < v0num) {
		Vec2u tmp = vtx[src];
		if (tmp[dir] < sep) {
			src++;
			continue;
		}
		if (tmp[dir] > sep) {
			vtx[src] = vtx[v1off];
			vtx[v1off] = tmp;
			v1off++;
		} else {
			vtx[src] = vtx[vsoff];
			vtx[vsoff] = tmp;
			vsoff++;
		}
	}
	src = v1off;
	while (src < v0num + v1num) {
		Vec2u tmp = vtx[src];
		if (tmp[dir] > sep) {
			src++;
			continue;
		} else {
			vtx[src] = vtx[vsoff];
			vtx[vsoff] = tmp;
			vsoff++;
		}
	}

	dissect_rect(vtx, v0min, v0max);
	dissect_rect(vtx + v0num, v1min, v1max);
}

static void load_face_interior(uint32_t subdiv, TArray<Vec2u> &F)
{
	assert(subdiv > 1);
	uint32_t ni = subdiv - 1;
	F.resize(ni * ni);
	for (uint32_t i = 1; i <= ni; ++i) {
		for (uint32_t j = 1; j <= ni; ++j) {
			F[ni * (i - 1) + (j - 1)] = (Vec2u){ i, j };
		}
	}
}

static void reorder_face_interior(uint32_t subdiv, TArray<Vec2u> &F)
{
	Vec2u min = { 1, 1 };
	uint32_t ni = subdiv - 1;
	Vec2u max = { ni, ni };
	dissect_rect(F.data, min, max);
}

#define LOAD_VTX(_offset, _x, _y, _z)                      \
	do {                                               \
		size_t offset = _offset;                   \
		size_t key = N * (N * (_z) + (_y)) + (_x); \
		pos[offset].x = 2 * (float)(_x) / n - 1;   \
		pos[offset].y = 2 * (float)(_y) / n - 1;   \
		pos[offset].z = 2 * (float)(_z) / n - 1;   \
		T.set_at(key, offset);                     \
	} while (0)

static void load_cube_vertices(Vec3 *__restrict pos, size_t subdiv,
			       GridTable &T, const TArray<Vec2u> &F)
{
	uint32_t ni = subdiv - 1;
	uint32_t iface_count = ni * ni;

	size_t foff[6];
	for (int f = 0; f < 6; ++f) {
		foff[f] = f * iface_count;
	}

	size_t n = subdiv;
	size_t N = n + 1;
	/* Load 6 face interiors */
	for (size_t k = 0; k < iface_count; ++k) {
		uint32_t x = F[k].x;
		uint32_t y = F[k].y;
		LOAD_VTX(foff[0]++, x, 0, y); /* Front  */
		LOAD_VTX(foff[1]++, n, x, y); /* Right  */
		LOAD_VTX(foff[2]++, n - x, n, y); /* Back   */
		LOAD_VTX(foff[3]++, 0, n - x, y); /* Left   */
		LOAD_VTX(foff[4]++, x, n - y, 0); /* Bottom */
		LOAD_VTX(foff[5]++, x, y, n); /* Top    */
	}
	/* Load 12 edge interiors */
	size_t eoff[12];
	for (int e = 0; e < 12; ++e) {
		eoff[e] = foff[5] + e * ni;
	}
	for (size_t k = 1; k <= ni; ++k) {
		/* Bottom 4 */
		LOAD_VTX(eoff[0]++, k, 0, 0);
		LOAD_VTX(eoff[1]++, n, k, 0);
		LOAD_VTX(eoff[2]++, n - k, n, 0);
		LOAD_VTX(eoff[3]++, 0, n - k, 0);
		/* Top 4 */
		LOAD_VTX(eoff[4]++, k, 0, n);
		LOAD_VTX(eoff[5]++, n, k, n);
		LOAD_VTX(eoff[6]++, n - k, n, n);
		LOAD_VTX(eoff[7]++, 0, n - k, n);
		/* Vert 4 */
		LOAD_VTX(eoff[8]++, 0, 0, k);
		LOAD_VTX(eoff[9]++, n, 0, k);
		LOAD_VTX(eoff[10]++, n, n, k);
		LOAD_VTX(eoff[11]++, 0, n, k);
	}
	size_t coff = eoff[11];
	/* Load 8 corners */
	LOAD_VTX(coff++, 0, 0, 0);
	LOAD_VTX(coff++, 0, n, 0);
	LOAD_VTX(coff++, n, n, 0);
	LOAD_VTX(coff++, n, 0, 0);
	LOAD_VTX(coff++, 0, 0, n);
	LOAD_VTX(coff++, 0, n, n);
	LOAD_VTX(coff++, n, n, n);
	LOAD_VTX(coff++, n, 0, n);

	assert(coff == 6 * subdiv * subdiv + 2);
}

#define GET_IDX(_x, _y, _z) (*T.get(N * (N * (_z) + (_y)) + (_x)))
#define LOAD_2TRI(f, a, b, c, d)    \
	do {                        \
		idx[foff[f]++] = a; \
		idx[foff[f]++] = b; \
		idx[foff[f]++] = c; \
		idx[foff[f]++] = a; \
		idx[foff[f]++] = c; \
		idx[foff[f]++] = d; \
	} while (0)

static void load_cube_indices(uint32_t *idx, size_t subdiv, const GridTable &T)
{
	size_t n = subdiv;
	size_t N = n + 1;
	size_t face_idx_count = 6 * n * n;

	size_t foff[6];
	for (int f = 0; f < 6; ++f) {
		foff[f] = f * face_idx_count;
	}

	for (uint32_t y = 0; y < n; ++y) {
		for (uint32_t x = 0; x < n; ++x) {
			uint32_t a, b, c, d;
			/* Front */
			a = GET_IDX(x, 0, y);
			b = GET_IDX(x + 1, 0, y);
			c = GET_IDX(x + 1, 0, y + 1);
			d = GET_IDX(x, 0, y + 1);
			LOAD_2TRI(0, a, b, c, d);
			/* Right */
			a = GET_IDX(n, x, y);
			b = GET_IDX(n, x + 1, y);
			c = GET_IDX(n, x + 1, y + 1);
			d = GET_IDX(n, x, y + 1);
			LOAD_2TRI(1, a, b, c, d);
			/* Back */
			a = GET_IDX(n - x, n, y);
			b = GET_IDX(n - x - 1, n, y);
			c = GET_IDX(n - x - 1, n, y + 1);
			d = GET_IDX(n - x, n, y + 1);
			LOAD_2TRI(2, a, b, c, d);
			/* Left */
			a = GET_IDX(0, n - x, y);
			b = GET_IDX(0, n - x - 1, y);
			c = GET_IDX(0, n - x - 1, y + 1);
			d = GET_IDX(0, n - x, y + 1);
			LOAD_2TRI(3, a, b, c, d);
			/* Bottom */
			a = GET_IDX(x, n - y, 0);
			b = GET_IDX(x + 1, n - y, 0);
			c = GET_IDX(x + 1, n - y - 1, 0);
			d = GET_IDX(x, n - y - 1, 0);
			LOAD_2TRI(4, a, b, c, d);
			/* Top */
			a = GET_IDX(x, y, n);
			b = GET_IDX(x + 1, y, n);
			c = GET_IDX(x + 1, y + 1, n);
			d = GET_IDX(x, y + 1, n);
			LOAD_2TRI(5, a, b, c, d);
		}
	}
}

int load_cube_nested_dissect(Mesh &m, size_t subdiv)
{
	if (subdiv <= 0 || subdiv > (1 << 14) /* 16K */) {
		return (-1);
	}

	size_t total_vtx = 6 * subdiv * subdiv + 2;
	size_t total_idx = 36 * subdiv * subdiv;

	TArray<Vec2u> F;
	load_face_interior(subdiv, F);
	reorder_face_interior(subdiv, F);

	GridTable T(total_vtx);

	/* Reserve vertices and indices */
	m.positions.resize(total_vtx);
	m.indices.resize(total_idx);

	/* First build vertices as six unattached faces of n^2 vertices each */
	load_cube_vertices(m.positions.data, subdiv, T, F);

	/* Build corresponding triangulation indices */
	load_cube_indices(m.indices.data, subdiv, T);
	return (0);
}

#undef LOAD_VTX
#undef GET_IDX
#undef LOAD_2TRI
