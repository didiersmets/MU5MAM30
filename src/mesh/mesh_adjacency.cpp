#include <string.h>

#include "mesh.h"

void compute_vertex_degrees(const Mesh &m, TArray<uint32_t> degrees)
{
	size_t vertex_count = m.vertex_count();
	size_t index_count = m.index_count();
	degrees.resize(vertex_count);
	memset(degrees.data, 0, vertex_count * sizeof(uint32_t));

	for (size_t i = 0; i < index_count; ++i) {
		degrees[m.indices[i]]++;
	}
}

