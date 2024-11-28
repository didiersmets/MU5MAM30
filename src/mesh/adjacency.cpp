#include "mesh.h"
#include "adjacency.h"

VTAdjacency::VTAdjacency(const Mesh &m)
{
	size_t vtx_count = m.vertex_count();
	size_t tri_count = m.triangle_count();
	size_t idx_count = 3 * tri_count;
	degree.resize(vtx_count);
	offset.resize(vtx_count);
	vtri.resize(idx_count);

	for (size_t i = 0; i < vtx_count; ++i) {
		degree[i] = 0;
		offset[i] = 0;
	}

	for (size_t i = 0; i < idx_count; ++i) {
		degree[m.indices[i]]++;
	}

	for (size_t v = 1; v < vtx_count; ++v) {
		offset[v] = offset[v - 1] + degree[v - 1];
	}

	for (size_t t = 0; t < tri_count; ++t) {
		uint32_t a = m.indices[3 * t + 0];
		uint32_t b = m.indices[3 * t + 1];
		uint32_t c = m.indices[3 * t + 2];
		vtri[offset[a]++] = { b, c };
		vtri[offset[b]++] = { c, a };
		vtri[offset[c]++] = { a, b };
	}

	for (size_t v = 0; v < vtx_count; ++v) {
		offset[v] -= degree[v];
	}
	assert(offset[vtx_count - 1] + degree[vtx_count - 1] == idx_count);
}
