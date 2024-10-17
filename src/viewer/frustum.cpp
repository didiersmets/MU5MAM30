#include <assert.h>

#include "ndc.h" /* reversed_y, reversed_z, z_zero_one */

#include "frustum.h"
#include "mat4.h"

Mat4 projection_matrix(const CameraFrustum& frustum)
{
	if (!frustum.is_ortho)
	{
		return perspective_matrix(
				frustum.aspect_x, frustum.aspect_y,
				frustum.shift_x, frustum.shift_y,
				frustum.near, frustum.far);
	}
	else 
	{
		return orthographic_matrix(
				frustum.aspect_x, frustum.aspect_y,
				frustum.shift_x, frustum.shift_y,
				frustum.near, frustum.far);
	}
}

Mat4 projection_matrix_inv(const CameraFrustum& frustum)
{
	if (!frustum.is_ortho)
	{
		return perspective_matrix_inv(
				frustum.aspect_x, frustum.aspect_y,
				frustum.shift_x, frustum.shift_y,
				frustum.near, frustum.far);
	}
	else 
	{
		return orthographic_matrix_inv(
				frustum.aspect_x, frustum.aspect_y,
				frustum.shift_x, frustum.shift_y,
				frustum.near, frustum.far);
	}
}

Mat4 perspective_matrix(float ax, float ay, float sx, float sy, float n, float f)
{
	assert(n != f);
	
	Mat4 M;

	/* col 0 */
	M(0,0) = ax;
	M(1,0) = 0.f;
	M(2,0) = 0.f;
	M(3,0) = 0.f;

	/* col 1 */
	M(0,1) = 0.f;
	if constexpr( reversed_y) M(1,1) = -ay;
	if constexpr(!reversed_y) M(1,1) = +ay;
	M(2,1) = 0.f;
	M(3,1) = 0.f;

	/* col 2*/
	M(0,2) = sx;
	if constexpr( reversed_y) M(1,2) = -sy;
	if constexpr(!reversed_y) M(1,2) = +sy;
	if constexpr( z_zero_one &&  reversed_z) 
		M(2,2) = n / (f - n); 
	if constexpr( z_zero_one && !reversed_z) 
		M(2,2) = -1.f / (1.f - n / f);
	if constexpr(!z_zero_one &&  reversed_z) 
		M(2,2) = + (1.f + n / f) / (1.f - n / f);
	if constexpr(!z_zero_one && !reversed_z) 
		M(2,2) = - (1.f + n / f) / (1.f - n / f);
	M(3,2) = -1.f;

	/* col 3 */
	M(0,3) = 0.f;
	M(1,3) = 0.f;
	if constexpr( z_zero_one &&  reversed_z) 
		M(2,3) = n / (1.f - n / f);
	if constexpr( z_zero_one && !reversed_z) 
		M(2,3) = -n / (1.f - n / f);
	if constexpr(!z_zero_one &&  reversed_z) 
		M(2,3) = 2.f * n / (1.f - n / f);
	if constexpr(!z_zero_one && !reversed_z) 
		M(2,3) = -2.f * n / (1.f - n / f);
	M(3,3) = 0.f;

	return (M);
}

Mat4 perspective_matrix_inv(float ax, float ay, float sx, float sy, float n, float f)
{
	
	assert(n != 0 && f != 0);

	const float inv_ax = 1.f / ax;
	const float inv_ay = 1.f / ay;
	const float inv_n = 1.f / n;
	const float inv_f = 1.f / f;

	Mat4 M;

	/* col 0 */
	M(0,0) = inv_ax;
	M(1,0) = 0.f;
	M(2,0) = 0.f;
	M(3,0) = 0.f;

	/* col 1 */
	M(0,1) = 0.f;
	if constexpr( reversed_y) M(1,1) = -inv_ay;
	if constexpr(!reversed_y) M(1,1) = inv_ay;
	M(2,1) = 0.f;
	M(3,1) = 0.f;

	/* col 2*/
	M(0,2) = 0.f;
	M(1,2) = 0.f;
	M(2,2) = 0.f;
	if constexpr( z_zero_one &&  reversed_z) 
		M(3,2) = inv_n - inv_f;
	if constexpr( z_zero_one && !reversed_z) 
		M(3,2) = inv_f - inv_n;
	if constexpr(!z_zero_one &&  reversed_z) 
		M(3,2) = 0.5f * (inv_n - inv_f);
	if constexpr(!z_zero_one && !reversed_z) 
		M(3,2) = 0.5f * (inv_f - inv_n);

	/* col 3 */
	M(0,3) = sx * inv_ax;
	if constexpr( reversed_y) M(1,3) = -sy * inv_ay;
	if constexpr(!reversed_y) M(1,3) = +sy * inv_ay;
	M(2,3) = -1.f;
	if constexpr( z_zero_one &&  reversed_z) 
		M(3,3) = inv_f;
	if constexpr( z_zero_one && !reversed_z) 
		M(3,3) = inv_n;
	if constexpr(!z_zero_one &&  reversed_z) 
		M(3,3) = 0.5 * (inv_n + inv_f);
	if constexpr(!z_zero_one && !reversed_z) 
		M(3,3) = 0.5 * (inv_n + inv_f);

	return (M);
}

Mat4 orthographic_matrix(float ax, float ay, float sx, float sy, float n, float f)
{
	assert(n != f);

	Mat4 M;

	/* col 0 */
	M(0,0) = ax;
	M(1,0) = 0.f;
	M(2,0) = 0.f;
	M(3,0) = 0.f;

	/* col 1 */
	M(0,1) = 0.f;
	if constexpr( reversed_y) M(1,1) = -ay;
	if constexpr(!reversed_y) M(1,1) = +ay;
	M(2,1) = 0.f;
	M(3,1) = 0.f;

	/* col 2*/
	M(0,2) = 0.f;
	M(1,2) = 0.f;
	if constexpr( z_zero_one &&  reversed_z) 
		M(2,2) = 1.0f / (f - n);
	if constexpr( z_zero_one && !reversed_z) 
		M(2,2) = -1.0f / (f - n);
	if constexpr(!z_zero_one &&  reversed_z) 
		M(2,2) = 2.0f / (f - n);
	if constexpr(!z_zero_one && !reversed_z) 
		M(2,2) = -2.0f / (f - n);
	M(3,2) = 0.f;

	/* col 3 */
	M(0,3) = -sx;
	if constexpr( reversed_y) M(1,3) = +sy;
	if constexpr(!reversed_y) M(1,3) = -sy;
	if constexpr( z_zero_one &&  reversed_z) 
		M(2,3) = 1.0f / (1.0f - n / f);
	if constexpr( z_zero_one && !reversed_z) 
		M(2,3) = -n / (f - n);
	if constexpr(!z_zero_one &&  reversed_z) 
		M(2,3) = (1.0f + n / f) / (1.0f - n / f);
	if constexpr(!z_zero_one && !reversed_z) 
		M(2,3) = -(1.0f + n / f) / (1.0f - n / f);
	M(3,3) = 1.f;
	
	return (M);
}

Mat4 orthographic_matrix_inv(float ax, float ay, float sx, float sy, float n, float f)
{
	assert(n != f);

	const float inv_ax = 1.f / ax;
	const float inv_ay = 1.f / ay;

	Mat4 M;

	/* col 0 */
	M(0,0) = inv_ax;
	M(1,0) = 0.f;
	M(2,0) = 0.f;
	M(3,0) = 0.f;

	/* col 1 */
	M(0,1) = 0.f;
	if constexpr( reversed_y) M(1,1) = -inv_ay;
	if constexpr(!reversed_y) M(1,1) = +inv_ay;
	M(2,1) = 0.f;
	M(3,1) = 0.f;

	/* col 2*/
	M(0,2) = 0.f;
	M(1,2) = 0.f;
	if constexpr( z_zero_one &&  reversed_z) 
		M(2,2) = f - n;
	if constexpr( z_zero_one && !reversed_z) 
		M(2,2) = n - f;
	if constexpr(!z_zero_one &&  reversed_z) 
		M(2,2) = 0.5f * (f - n);
	if constexpr(!z_zero_one && !reversed_z) 
		M(2,2) = 0.5f * (n - f);
	M(3,2) = 0.f;

	/* col 3 */
	M(0,3) = sx * inv_ax;
	if constexpr( reversed_y) M(1,3) = -sy * inv_ay;
	if constexpr(!reversed_y) M(1,3) = +sy * inv_ay;
	if constexpr( z_zero_one &&  reversed_z) 
		M(2,3) = -f;
	if constexpr( z_zero_one && !reversed_z) 
		M(2,3) = -n;
	if constexpr(!z_zero_one &&  reversed_z) 
		M(2,3) = -0.5f * (n + f);
	if constexpr(!z_zero_one && !reversed_z) 
		M(2,3) = -0.5f * (n + f);
	M(3,3) = 1.f;
	
	return (M);
}
