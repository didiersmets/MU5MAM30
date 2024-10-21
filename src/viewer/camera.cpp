#include <assert.h>
#include <math.h>

#include "aabb.h"
#include "camera.h"
#include "geometry.h"
#include "mat4.h"
#include "ndc.h"
#include "quat.h"
#include "transform.h"
#include "vec3.h"

Camera::Camera(float aspect_ratio, float fov, Fov axis)

{
	assert(aspect_ratio > 0.f && fov > 0.f);

	float focal_ratio = 1.f / tan(fov * DEG2RAD * 0.5f);

	if (axis == Horizontal) {
		frustum.aspect_x = focal_ratio;
		frustum.aspect_y = focal_ratio * aspect_ratio;

	} else {
		frustum.aspect_x = focal_ratio / aspect_ratio;
		frustum.aspect_y = focal_ratio;
	}
}

Camera &Camera::set_aspect(float aspect_ratio, const Fov cst_axis)
{
	assert(aspect_ratio > 0);

	if (cst_axis == Horizontal) {
		frustum.aspect_y = frustum.aspect_x * aspect_ratio;
	} else {
		frustum.aspect_x = frustum.aspect_y / aspect_ratio;
	}
	return (*this);
}

Camera &Camera::set_fov(float fov, Fov axis)
{
	this->clip_to_world();
	float aspect_ratio = frustum.aspect_y / frustum.aspect_x;
	float focal_ratio = 1.f / tan(fov * DEG2RAD * 0.5f);

	if (axis == Horizontal) {
		frustum.aspect_x = focal_ratio;
		frustum.aspect_y = focal_ratio * aspect_ratio;

	} else {
		frustum.aspect_x = focal_ratio / aspect_ratio;
		frustum.aspect_y = focal_ratio;
	}
	return (*this);
}

Camera &Camera::set_lense_shift(float shift_x, float shift_y)
{
	frustum.shift_x = shift_x;
	frustum.shift_y = shift_y;
	return (*this);
}

Camera &Camera::set_orthographic(bool is_ortho)
{
	frustum.is_ortho = is_ortho;
	return (*this);
}

Vec3 Camera::get_position() const
{
	return (position);
}
Quat Camera::get_rotation() const
{
	return rotation;
}
Vec3 Camera::get_target() const
{
	return target;
}

Camera &Camera::set_position(const Vec3 &position)
{
	this->position = position;
	return (*this);
}

Camera &Camera::set_rotation(const Quat &rotation)
{
	this->rotation = rotation;
	return (*this);
}

Camera &Camera::set_target(const Vec3 &target)
{
	this->target = target;
	return (*this);
}

Camera &Camera::translate(const Vec3 &t, Space coord)
{
	if (coord == View) {
		position += ::rotate(t, rotation);
	} else /* World */
	{
		position += t;
	}
	return (*this);
}

Camera &Camera::rotate(const Quat &r, Space coord)
{
	assert(!approx_equal(norm(r), 0.f));
	Quat rot = r * (1.f / norm(r));
	if (coord == View) {
		rot.xyz = ::rotate(rot.xyz, rotation);
	}
	rotation = compose(rotation, rot);
	return (*this);
}

Camera &Camera::orbit(const Quat &r, Space coord)
{
	assert(!approx_equal(norm(r), 0.f));
	Quat rot = r * (1.f / norm(r));
	if (coord == View) {
		rot.xyz = ::rotate(rot.xyz, rotation);
	}
	rotation = compose(rotation, rot).normalise();
	position = ::orbit(position, rot, target);
	return (*this);
}

Camera &Camera::zoom(float factor)
{
	Vec3 new_pos = target + (1.f / factor) * (position - target);
	set_position(new_pos);
	return (*this);
}

float Camera::get_near() const
{
	return frustum.near;
}

float Camera::get_far() const
{
	return frustum.far;
}

Camera &Camera::set_near(float near_plane)
{
	frustum.near = near_plane;
	return (*this);
}

Camera &Camera::set_far(float far_plane)
{
	frustum.far = far_plane;
	return (*this);
}

Mat4 Camera::view_to_clip() const
{
	return projection_matrix(frustum);
}

Mat4 Camera::clip_to_view() const
{
	return projection_matrix_inv(frustum);
}

Mat4 Camera::world_to_view() const
{
	return compose(-position, -rotation).as_matrix();
}

Mat4 Camera::view_to_world() const
{
	return compose(rotation, position).as_matrix();
}

Mat4 Camera::world_to_clip() const
{
	return compose(world_to_view(), view_to_clip());
}

Mat4 Camera::clip_to_world() const
{
	return compose(clip_to_view(), view_to_world());
}

Ray Camera::view_ray_at(float x, float y) const
{
	assert(0.f <= x && x <= 1.f && 0.f <= y && y <= 1.f);

	Vec3 ndc = nwd_to_ndc(x, y, 0.5f);
	Vec3 v = transform(clip_to_view(), ndc);

	return { .start = Vec3::Zero, .dir = v };
}

Ray Camera::world_ray_at(float x, float y) const
{
	assert(0.f <= x && x <= 1.f && 0.f <= y && y <= 1.f);

	Vec3 ndc = nwd_to_ndc(x, y, 0.5f);
	Vec3 v = transform(clip_to_world(), ndc);

	return { .start = position, .dir = v };
}

Vec3 Camera::view_coord_at(float x, float y, float depth) const
{
	assert(0.f <= x && x <= 1.f && 0.f <= y && y <= 1.f);
	assert(0.f <= depth && depth <= 1.f);

	Vec3 ndc = nwd_to_ndc(x, y, depth);

	return transform(clip_to_view(), ndc);
}

Vec3 Camera::world_coord_at(float x, float y, float depth) const
{
	assert(0.f <= x && x <= 1.f && 0.f <= y && y <= 1.f);
	assert(0.f <= depth && depth <= 1.f);

	Vec3 ndc = nwd_to_ndc(x, y, depth);

	return transform(clip_to_world(), ndc);
}

void Camera::save_spatial_state()
{
	saved_rotation = rotation;
	saved_position = position;
}

void Camera::restore_spatial_state()
{
	rotation = saved_rotation;
	position = saved_position;
}

int is_visible(const float *vtx, int n, const float *pvm)
{
	bool clipx, clip_R, clip_L;
	bool clipy, clip_T, clip_B;
	bool clipz, clip_F, clip_N;
	float T, W;
	clipx = clip_R = clip_L = 1;
	clipy = clip_T = clip_B = 1;
	clipz = clip_F = clip_N = 1;
	for (int i = 0; i < n; ++i) {
		W = pvm[3] * vtx[3 * i + 0] + pvm[7] * vtx[3 * i + 1] +
		    pvm[11] * vtx[3 * i + 2] + pvm[15];
		if (clipx) {
			T = pvm[0] * vtx[3 * i + 0] + pvm[4] * vtx[3 * i + 1] +
			    pvm[8] * vtx[3 * i + 2] + pvm[12];
			clip_R = clip_R && (T > W);
			clip_L = clip_L && (T < -W);
			clipx = clip_R || clip_L;
		}
		if (clipy) {
			T = pvm[1] * vtx[3 * i + 0] + pvm[5] * vtx[3 * i + 1] +
			    pvm[9] * vtx[3 * i + 2] + pvm[13];
			clip_T = clip_T && (T > W);
			clip_B = clip_B && (T < -W);
			clipy = clip_T || clip_B;
		}
		if (clipz) {
			T = pvm[2] * vtx[3 * i + 0] + pvm[6] * vtx[3 * i + 1] +
			    pvm[10] * vtx[3 * i + 2] + pvm[14];
			clip_F = clip_F && (T > W);
			clip_N = clip_N && (T < -W);
			clipz = clip_F || clip_N;
		}
		if (!(clipx || clipy || clipz))
			return 1;
	}
	return 0;
}

enum Visibility visibility(const Aabb &bbox, const float *pvm)
{
	bool clip_R = true;
	bool clip_L = true;
	bool clip_T = true;
	bool clip_B = true;
	bool clip_F = true;
	bool clip_N = true;

	bool fully_in = true;

	float T, W;

	for (int i = 0; i < 8; ++i) {
		float x = (i & 1) ? (bbox.min.x) : (bbox.max.x);
		float y = (i & 2) ? (bbox.min.y) : (bbox.max.y);
		float z = (i & 4) ? (bbox.min.z) : (bbox.max.z);

		W = pvm[3] * x + pvm[7] * y + pvm[11] * z + pvm[15];
		T = pvm[0] * x + pvm[4] * y + pvm[8] * z + pvm[12];
		bool is_R = (T > W);
		bool is_L = (T < -W);
		clip_R = clip_R && is_R;
		clip_L = clip_L && is_L;
		fully_in = fully_in && !is_R && !is_L;

		T = pvm[1] * x + pvm[5] * y + pvm[9] * z + pvm[13];
		bool is_T = (T > W);
		bool is_B = (T < -W);
		clip_T = clip_T && is_T;
		clip_B = clip_B && is_B;
		fully_in = fully_in && !is_T && !is_B;

		T = pvm[2] * x + pvm[6] * y + pvm[10] * z + pvm[14];
		bool is_F = (T > W);
		bool is_N = (T < -W);
		clip_F = clip_F && is_F;
		clip_N = clip_N && is_N;
		fully_in = fully_in && !is_F && !is_N;
	}

	if (clip_R || clip_L || clip_T || clip_B || clip_F || clip_N)
		return Visibility::None;
	return fully_in ? Visibility::Full : Visibility::Partial;
}

