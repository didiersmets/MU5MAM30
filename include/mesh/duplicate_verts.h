#pragma once

#include <stddef.h>
#include <stdint.h>

#include "mesh.h"
#include "vec3.h"

/* @param (IN) pos   : pointer to vertices
 * @param (IN) count : number of vertices
 * @param (OUT)remap : pointer to remaping array (must be pre-allocated for
 *                     pos_count uint32_t)

 * The function fills the array remap with the property that
 *          remap[i] = remap[j] iff pos[i] == pos[j]
 *          remap[i] <= i
 * In other words, after completion remap[i] contains the smallest index
 * 0 <= j <= i s.t. pos[j] = pos[i].
 *
 * Note : remap must be pre-allocated for pos_count uint32_t.
 *
 * Return value : the number of unique vertices (obviously <= count)
 *
 */
size_t build_position_remap(const Vec3 *pos, size_t count, uint32_t *remap);

/* Builds the position remap, rebuild the mesh positions array according to
 * that remap, i.e.
 *     new_pos[remap[i]] = old_pos[i] for all 0 <= i < old_vertex_count
 *     new_idx[j] = remap[old_idx[j]] for all 0 <= j < index_count
 * After completion, the size of m.positions is set to new_vertex_count.
 */
void remove_duplicate_vertices(Mesh &m);
