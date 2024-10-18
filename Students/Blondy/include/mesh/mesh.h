#pragma once
#include <stdint.h>

#include "array.h"
#include "vec3.h"

struct Mesh {
	TArray<Vec3> positions;
	TArray<uint32_t> indices;
	size_t vertex_count() const
	{
		return positions.size;
	}
	size_t index_count() const
	{
		return indices.size;
	}
	size_t triangle_count() const
	{
		return indices.size / 3;
	}
};
