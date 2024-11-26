#include <stdint.h>

#include "array.h"
#include "mesh.h"

/* Vertex to Triangle adjacency table */
struct VTAdjacency {
	struct VTri {
		uint32_t next;
		uint32_t prev;
	};
	TArray<uint32_t> degree;
	TArray<uint32_t> offset;
	TArray<VTri> vtri;
	VTAdjacency(const Mesh &m);
};
