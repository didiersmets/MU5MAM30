#include <stdint.h>
#include <stdio.h>

#include "cube.h"
#include "mesh.h"
#include "vec3.h"

int load_sphere(Mesh &m, size_t subdiv)
{
	if (int res = load_cube(m, subdiv))
		return (res);

	Vec3 *pos = m.positions.data;
	size_t vtx_count = m.positions.size;
	for (size_t i = 0; i < vtx_count; ++i) {
		pos[i] = normalized(pos[i]);
	}

	return (0);
}
