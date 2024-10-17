#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES 1
#endif
#include <GL/gl.h>
#include <GL/glext.h>

#include "imgui/imgui.h"

#include "viewer_app.h"

#include "chrono.h"
#include "camera.h"
#include "cube.h"
#include "mesh_bounds.h"
#include "mesh_io.h"
#include "shaders.h"
#include "sphere.h"
#include "trackball.h"
#include "transform.h"

#define ZOOM_SENSITIVITY 0.3f /* for mouse wheel zoom */
#define ORBIT_SENSITIVITY 1.f

void syntax(char *prg_name)
{
	printf("Syntax : %s ($(obj_filename)| cube | sphere) [n]", prg_name);
	printf("         Subdivision number n must be provided in case of "
	       "         cube or sphere mesh.\n");
	exit(EXIT_FAILURE);
}

ViewerApp::ViewerApp(int argc, char **argv)
	: GUIApp("Viewer")
{
	Timer chrono;
	chrono.start();

	int res = -1;
	if (argc > 2 && strncmp(argv[1], "cube", 4) == 0) {
		res = load_cube(mesh, atoi(argv[2]));
	} else if (argc > 2 && strncmp(argv[1], "sphere", 5) == 0) {
		res = load_sphere(mesh, atoi(argv[2]));
	} else if (argc > 1) {
		res = load_obj(argv[1], mesh);
	}
	if (res)
		syntax(argv[0]);

	printf("Vertices : %zu. Triangles : %zu\n", mesh.vertex_count(),
	       mesh.triangle_count());
	chrono.stop("loading mesh");

	Aabb bbox = compute_mesh_bounds(mesh);
	Vec3 model_center = (bbox.min + bbox.max) * 0.5f;
	Vec3 model_extent = (bbox.max - bbox.min);
	float model_size = max(model_extent);
	if (model_size == 0) {
		printf("Warning : Mesh is empty or reduced to a point.\n");
		model_size = 1;
	}
	target = model_center;
	Vec3 start_pos = (model_center + 2.f * Vec3(0, 0, model_size));

	camera.set_aspect(float(width) / height);
	camera.set_fov(45);
	camera.set_position(start_pos);
	camera.set_near(0.01 * model_size);
	camera.set_far(100 * model_size);

	draw_surface = true;
	draw_edges = true;
}

void ViewerApp::opengl_init()
{
	GUIApp::opengl_init();

	shader = create_shader("./shaders/default.vert",
			       "./shaders/default.frag");
	if (!shader)
		exit(EXIT_FAILURE);

	gpu_mesh.m = &mesh;
	gpu_mesh.upload();
}

void ViewerApp::opengl_fini()
{
	glDeleteProgram(shader);

	GUIApp::opengl_fini();
}

void ViewerApp::mouse_button_callback(int button, int action, int mods)
{
	GUIApp::mouse_button_callback(button, action, mods);

	if (button == 0 && action == GLFW_PRESS) {
		float px = mouse.x;
		float py = mouse.y;
		last_camera_pos = camera.get_position();
		last_camera_rot = camera.get_rotation();
		last_trackball_v = screen_trackball(px, py, width, height);

		if (!mouse.is_double_click[0])
			return;

		float depth;
		float tx = px;
		float ty = height - py;
		glReadPixels(tx, ty, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT,
			     &depth);

		/* Don't track clicks outisde of model */
		if (approx_equal(depth, 1.f)) {
			return;
		}
		target = camera.world_coord_at(px / width, py / height, depth);
	}
}

void ViewerApp::cursor_pos_callback(double px, double py)
{
	GUIApp::cursor_pos_callback(px, py);

	if (!mouse.is_pressed[0])
		return;

	int mods = mouse.mods[0];
	float last_click_x = mouse.last_click_x[0];
	float last_click_y = mouse.last_click_y[0];

	if (mods & GLFW_MOD_SHIFT) {
		/* Translation in x and y */
		float dist = norm(target - camera.get_position());
		float mult = dist / width;
		Vec3 trans;
		trans.x = (last_click_x - px) * mult;
		trans.y = (py - last_click_y) * mult;
		trans.z = 0.f;
		camera.set_position(last_camera_pos);
		camera.translate(trans, Camera::View);
	} else if (mods & GLFW_MOD_CONTROL) {
		/* Zoom (translation in target direction) */
		float mu = ZOOM_SENSITIVITY * (px - last_click_x) / 100;
		Vec3 new_pos = target + exp(mu) * (last_camera_pos - target);
		camera.set_position(new_pos);
	} else /* no modifier */
	{
		Vec3 trackball_v = screen_trackball(px, py, width, height);
		Quat rot = great_circle_rotation(last_trackball_v, trackball_v);
		/* rot quat is in view frame, back to world frame */
		rot.xyz = rotate(rot.xyz, last_camera_rot);
		camera.set_position(last_camera_pos);
		camera.set_rotation(last_camera_rot);
		/* TODO adapt sensitivity to camera fov in free mode*/
		rot = pow(rot, ORBIT_SENSITIVITY);
		camera.orbit(-rot, target);
		/**
		 * Great_circle_rotation is singular when from and to are
		 * close to antipodal. To avoid that situation, we checkout
		 * mouse move when the from and to first belong to two opposite
		 * hemispheres.
		 */
		if (dot(trackball_v, last_trackball_v) < 0) {
			last_click_x = px;
			last_click_y = py;
			last_camera_pos = camera.get_position();
			last_camera_rot = camera.get_rotation();
			last_trackball_v = trackball_v;
		}
	}
}

void ViewerApp::scroll_callback(double xoffset, double yoffset)
{
	(void)xoffset;
	Vec3 old_pos = camera.get_position();
	Vec3 new_pos = target + exp(-ZOOM_SENSITIVITY * (float)yoffset) *
					(old_pos - target);
	camera.set_position(new_pos);
}

void ViewerApp::key_callback(int key, int scancode, int action, int mods)
{
	(void)scancode;
	(void)mods;

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		stop();
		return;
	}
	if (key == GLFW_KEY_S && action == GLFW_PRESS) {
		draw_surface = !draw_surface;
		return;
	}
	if (key == GLFW_KEY_E && action == GLFW_PRESS) {
		draw_edges = !draw_edges;
		return;
	}
}

void ViewerApp::resize(int width, int height)
{
	GUIApp::resize(width, height);
	camera.set_aspect((float)width / height);
	glViewport(0, 0, width, height);
}

void ViewerApp::pre_draw()
{
	GUIApp::pre_draw();
	glUseProgram(shader);
	Mat4 proj = camera.view_to_clip();
	Mat4 vm = camera.world_to_view();
	Vec3 camera_pos = camera.get_position();
	glUniformMatrix4fv(0, 1, 0, &vm(0, 0));
	glUniformMatrix4fv(1, 1, 0, &proj(0, 0));
	glUniform3fv(2, 1, &camera_pos[0]);
}

void ViewerApp::draw_scene()
{
	if (draw_surface) {
		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(-1.f, -1.f);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glUniform1i(3, true);
		gpu_mesh.draw();
	}
	if (draw_edges) {
		glDisable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(0.f, 0.f);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glUniform1i(3, false);
		gpu_mesh.draw();
	}
}

void ViewerApp::draw_gui()
{
	ImGui::Begin("Controls");
	float fps = ImGui::GetIO().Framerate;
	ImGui::Text("Average %.3f ms/frame (%.1f FPS)", 1000.0f / fps, fps);

	ImGui::End();
}

