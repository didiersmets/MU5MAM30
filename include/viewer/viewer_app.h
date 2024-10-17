#pragma once

#include "camera.h"
#include "gui_app.h"
#include "mesh.h"
#include "mesh_gpu.h"
#include "mouse.h"

struct ViewerApp : public GUIApp {
	/* Methods */
	ViewerApp(int argc, char **argv);
	void mouse_button_callback(int button, int action, int mods) override;
	void cursor_pos_callback(double x, double y) override;
	void scroll_callback(double xoffset, double yoffset) override;
	void key_callback(int key, int scancode, int action, int mods) override;
	void opengl_init() override;
	void resize(int width, int height) override;
	void pre_draw() override;
	void draw_scene() override;
	void draw_gui() override;
	void opengl_fini() override;

	bool load_mesh(const char *filename);
	bool upload_mesh_to_gpu();
	bool update_gpu_mesh();

	/* Members */
	Camera camera;
	Mesh mesh;
	GPUMesh gpu_mesh;
	GLuint shader;
	Vec3 last_camera_pos;
	Quat last_camera_rot;
	Vec3 last_trackball_v;
	Vec3 target;
	bool draw_surface;
	bool draw_edges;
};
