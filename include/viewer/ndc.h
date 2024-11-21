#pragma once

#include "gl_utils.h"

#include "vec3.h"

#define NDC_REVERSED_Y 0
#ifndef __APPLE__
	#define NDC_REVERSED_Z 1
	#define NDC_Z_ZERO_ONE 1
#else
	#define NDC_REVERSED_Z 0
	#define NDC_Z_ZERO_ONE 0
#endif

/* The various graphic API do NOT agree on the definition
 * of Normalized Device Coordinates (hereafter NDC).
 * In the sequel :
 *	reversed_y  : origin for NDC.xy is top-left.
 *	reversed_z  : depth = 1 for near_plane (instead of far_plane).
 *	zero_to_one : NDC.z ranges from O to 1 (instead of [-1, 1]).
 * These are important when using the hardware rasterizer, since they
 * imply the exact form of the expected projection matrix.
 *
 * It is sometimes handy to use Normalized Window and Depth coordinates (NWD).
 * They always range in [0,1]^3.
 * For the xy plane, (0,0) is always at the top left corner of the window
 * (that's what all window managers do).
 * For the z plane, whether reversed_z holds or not is left undetermined
 * (and therefore functions using NWD coord must test it).
 * */

#if (!defined NDC_REVERSED_Y)
	#if (defined VULKAN)
		#warning NDC_REVERSED_Y undefined, assuming 1
		#define NDC_REVERSED_Y 1
	#else
		#warning NDC_REVERSED_Y undefined, assuming 0
		#define NDC_REVERSED_Y 0
	#endif
#endif

#if (!defined NDC_Z_ZERO_ONE)
	#if (defined VULKAN || defined METAL)
		#warning NDC_Z_ZERO_ONE undefined, assuming 1
		#define NDC_Z_ZERO_ONE 1
	#else
		#warning NDC_Z_ZERO_ONE undefined, assuming 0
		#define NDC_Z_ZERO_ONE 0
	#endif
#endif

#if (!defined NDC_REVERSED_Z)
	#if (!NDC_Z_ZERO_ONE)
		#warning NDC_REVERSED_Z undefined, assuming 0
		#define NDC_REVERSED_Z 0
	#else /* NDC_DEPTH_IS_ZERO_TO_ONE */
		#warning NDC_REVERSED_Z undefined, assuming 1
		#define NDC_REVERSED_Z 1
	#endif
#endif

constexpr bool reversed_y = NDC_REVERSED_Y ? true : false;
constexpr bool reversed_z = NDC_REVERSED_Z ? true : false;
constexpr bool z_zero_one = NDC_Z_ZERO_ONE ? true : false;

inline Vec3 nwd_to_ndc(float x, float y, float depth)
{
	Vec3 ndc{0, 0, 0};

	ndc.x = 2.f * x - 1.f;
	if constexpr (reversed_y)
		ndc.y = 2.f * y - 1.f;
	if constexpr (!reversed_y)
		ndc.y = 1.f - 2.f * y;
	if constexpr (z_zero_one)
		ndc.z = depth;
	if constexpr (!z_zero_one)
		ndc.z = 2.f * depth - 1.f;

	return (ndc);
}

inline void set_up_opengl_for_ndc()
{
#ifndef __APPLE__
	constexpr GLenum origin = reversed_y ? GL_UPPER_LEFT : GL_LOWER_LEFT;
	constexpr GLenum depth =
	    z_zero_one ? GL_ZERO_TO_ONE : GL_NEGATIVE_ONE_TO_ONE;
	glClipControl(origin, depth);
#endif
	if constexpr (reversed_z) {
		glDepthFunc(GL_GREATER);
		glClearDepth(0.0f);
	}
}

