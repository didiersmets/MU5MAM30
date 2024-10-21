#pragma once

#include "geometry.h"
#include "mat4.h"
/**
 * General frustum defined by six planes.
 */
struct Frustum {
	Plane l;
	Plane r;
	Plane t;
	Plane b;
	Plane n;
	Plane f;
};

/**
 * Camera Frustum
 *
 * A restricted class of frustum for which
 * l, r, t, b, n and f can be associated to scalar values.
 *
 * The physical parameters of interest in the camera are:
 *
 *	1) Focal Length  : L
 *	2) Sensor Width  : W
 *	3) Sensor Height : H
 *
 * Only the (unit less) ratios between them matter for rendering, and only
 * subsets of two of these ratios are independent. We choose:
 *
 *	aspect_x := L / W = 1 / (2 * tan(fov_x / 2)),
 *	aspect_y := L / W = 1 / (2 * tan(fov_y / 2)),
 *
 * because they are closely linked to projection matrices.
 *
 * The lense axis is assumed perpendicular to the sensor plane, but may be
 * shifted with respect to sensore center. In terms of frustum planes:
 *
 *	shift_x = (r+l)/(r-l),
 *	shift_y = (t+b)/(t-b).
 *
 * An additional option for (non physical) orthographic camera frustum
 * is added. In that case aspects and shifts have a to be reinterpreted.
 */
struct CameraFrustum {
	float aspect_x = 2.f;
	float aspect_y = 2.f;
	float shift_x = 0.f;
	float shift_y = 0.f;
	float near = 0.001f;
	float far = 100.f;
	bool is_ortho = false;
};

Mat4 projection_matrix(const CameraFrustum &f);
Mat4 projection_matrix_inv(const CameraFrustum &f);

Mat4 perspective_matrix(float ax, float ay, float sx, float sy, float n,
			float f);
Mat4 perspective_matrix_inv(float ax, float ay, float sx, float sy, float n,
			    float f);
Mat4 orthographic_matrix(float ax, float ay, float sx, float sy, float n,
			 float f);
Mat4 orthographic_matrix_inv(float ax, float ay, float sx, float sy, float n,
			     float f);
