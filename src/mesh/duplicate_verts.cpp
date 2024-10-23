#include <stddef.h>
#include <stdint.h>

#include "array.h"
#include "hash.h"
#include "hash_table.h"
#include "mesh.h"
#include "vec3.h"

struct PositionHasher {
	const Vec3 *pos;
	static constexpr uint32_t empty_key = ~static_cast<uint32_t>(0);
	size_t hash(uint32_t key) const
	{
		uint32_t hash = 0;
		const uint32_t *p =
		    reinterpret_cast<const uint32_t *>(pos + key);
		hash = murmur2_32(hash, p[0]);
		hash = murmur2_32(hash, p[1]);
		hash = murmur2_32(hash, p[2]);
		return hash;
	}

	bool is_empty(uint32_t key) const { return (key == empty_key); }
	bool is_equal(uint32_t key1, uint32_t key2) const
	{
		return pos[key1] == pos[key2];
	}
};

size_t build_position_remap(Vec3 *pos, size_t count, uint32_t *remap)
{
	PositionHasher hasher{pos};
	HashTable<uint32_t, uint32_t, PositionHasher> vtx_remap(count, hasher);

	size_t new_count = 0;
	for (size_t i = 0; i < count; ++i) {
		uint32_t *p = vtx_remap.get_or_set(i, new_count);
		if (p) {
			remap[i] = *p;
		} else {
			remap[i] = new_count;
			new_count++;
		}
	}
	return new_count;
}

void remove_duplicate_vertices(Mesh &m)
{
	Vec3 *pos = m.positions.data;
	size_t vtx_count = m.vertex_count();
	TArray<uint32_t> remap(m.vertex_count());

	size_t new_count = build_position_remap(pos, vtx_count, remap.data);

	/* Remap vertices */
	for (size_t i = 0; i < vtx_count; ++i) {
		assert(remap[i] <= i);
		pos[remap[i]] = pos[i];
	}
	m.positions.resize(new_count);

	/* Remap indices */
	size_t idx_count = m.index_count();
	uint32_t *idx = m.indices.data;
	for (size_t i = 0; i < idx_count; ++i) {
		idx[i] = remap[idx[i]];
	}
}
