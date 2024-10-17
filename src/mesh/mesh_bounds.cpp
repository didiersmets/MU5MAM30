#include "mesh_bounds.h"

#include "aabb.h"
#include "mesh.h"
#include "vec3.h"

Aabb compute_mesh_bounds(const Mesh &m)
{
	if (m.positions.size == 0) {
		return { Vec3::Zero, Vec3::Zero };
	}

	const Vec3 *positions = m.positions.data;
	size_t vertex_count = m.positions.size;

	Vec3 min = positions[0];
	Vec3 max = positions[0];

	for (size_t i = 1; i < vertex_count; ++i) {
		const Vec3 &pos = positions[i];

		for (size_t j = 0; j < 3; ++j) {
			min[j] = (pos[j] < min[j]) ? pos[j] : min[j];
			max[j] = (pos[j] > max[j]) ? pos[j] : max[j];
		}
	}

	return { min, max };
}
