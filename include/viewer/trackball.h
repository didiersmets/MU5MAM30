#pragma once

#include "camera.h"
#include "vec3.h"

struct ScreenTrackball {
	/* Methods */
	void grab(float px, float py, int width, int height);
	Quat drag(float px, float py, int width, int height);
	void release();
	void set_sensitivity(float sensitivity);
	/* Members */
	bool grabbed = false;
	Quat pre_rot;
	Vec3 last_v;
	float sensitivity = 1.f;
};

/* Transform a pixel coordinates into a point on the unit sphere,
 * modeling user click onto a trackball attached in front of the the screen .
 *
 * Implemented using (the inverse of) a stereographic projection from a sphere
 * of given radius and tangent to the screen plane at the screen center.
 *
 * NOTE: Screens coordinates are scaled so that the screen appears square and
 *       the sphere is tangent to the four screen sides.
 *
 * @param px - pixel x coord of click point (left is at 0).
 * @param py - pixel y coord of click point (top is at 0).
 * @param width - window width.
 * @param height - window height.
 *
 * Return : a unit 3D vector in a frame oriented like view coordinates (so for
 *          a click on the screen center the return vector is always (0, 0, 1).
 */
Vec3 screen_trackball(float px, float py, float width, float height);

/* Transform a normalised screen coordinate into a point on the unit sphere,
 * modeling user click onto a trackball attached to a given world position.
 *
 * Implemented using (the inverse of) a stereographic projection from a sphere
 * of given radius at given world position into a plane tangent to that sphere
 * and perpendicular to the view direction of the sphere center.
 * The screen coordinate it first projected to that screen using a camera
 * world ray.
 *
 * @param x - normalised screen coord of click point (left is 0 and right is 1).
 * @param y - normalised screen coord of click point (top is 0 and bottom is 1).
 * @param center - world coordinate of trackabll center.
 * @param radius - radius of trackball in world space.
 * @param camera - reference to the camera used for the screen to world mapping.
 *
 * Return : If intersection occurs, a unit 3D vector in world coordinates
 *          corresponding to the vector from trackball center to the trackball
 *          clicked point.
 *          If not, return Vec3::ZAxis.
 *
 * Can be used e.g. to model a manipulator in order to modify the model
 * matrix of a given mesh (the trackball in that case could be a bounding
 * sphere).
 */
Vec3 world_trackball(float x, float y, const Vec3 &center, float radius,
		     const Camera &camera);
