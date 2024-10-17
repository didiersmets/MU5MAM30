#include "trackball.h"
#include "vec3.h"
#include "quat.h"
#include "geometry.h"
#include "camera.h"

Vec3 screen_trackball(float px, float py, float width, float height)
{
	float x = (2 * px - width) / width;
	float y = (height - 2 * py) / height;
	float a = 2.f / (1.f + x * x + y * y);

	Vec3 v {a * x, a * y, -1.f + a};
	
	assert(approx_equal<float>(norm(v), 1.f));

	return v;
}

Vec3 world_trackball(float x, float y, const Vec3& center, float radius, 
			const Camera& camera)
{
	Vec3 view_dir = center - camera.get_position();
	float len = norm(view_dir);

	if (len == 0) 
	{
		return Vec3::ZAxis;
	}
	view_dir *= (1.f / len);

	Vec3 nearest = center - radius * view_dir;
	Plane tg_plane = plane_from_normal_and_point(view_dir, nearest);
	Ray ray = camera.world_ray_at(x, y);
	Vec4 test = ray_plane_intersection(ray, tg_plane);

	if (test.w == 0)
	{
		return Vec3::ZAxis;
	}

	Vec3 touch = test.xyz * (1.f / test.w);

	/* Stereographic projection, from the point diametrically
	 * opposite to nearest, and to the tangent plane at nearest.
	 * Computation : shoot ray from stereographic center ( = center + 
	 * radius * view_dir) towards touch, and stop when the distance 
	 * towards center becomes equal to radius. Since we shoot from the 
	 * sphere boundary there is no second order equation to solve, only 
	 * first order.
	 */
	Vec3 sph_dir = (touch - center) * (1.f / radius);
	float tmp = dot(view_dir, sph_dir);
	float s = 2.f * (tmp - 1) / (dot(sph_dir, sph_dir) - 2.f * tmp + 1.f);

	Vec3 v = view_dir + s * (view_dir - sph_dir);
	
	assert(approx_equal<float>(norm(v), 1.f));

	return v;
}
