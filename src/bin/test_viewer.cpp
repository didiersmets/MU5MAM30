#include <stdlib.h>
#include <string.h>

#ifndef GL_GLEXT_PROTOTYPES
	#define GL_GLEXT_PROTOTYPES 1
#endif
#include "imgui/imgui.h"
#include <GL/gl.h>
#include <GL/glext.h>

#include "chrono.h"
#include "cube.h"
#include "mesh.h"
#include "mesh_bounds.h"
#include "mesh_gpu.h"
#include "mesh_io.h"
#include "ndc.h"
#include "shaders.h"
#include "sphere.h"
#include "viewer.h"

struct Cfg {
	bool draw_surface = true;
	bool draw_edges = true;
	float bgcolor[4] = {0.3, 0.3, 0.3, 1.0};
} cfg;

static void syntax(char *prg_name);
static int load_mesh(Mesh &mesh, int argc, char **argv);
static void init_camera_for_mesh(const Mesh &mesh, Camera &camera);
static void update_all();
static void draw_scene(const Viewer &viewer, int shader,
		       const GPUMesh &gpu_mesh);
static void draw_gui();
static void key_cb(int key, int action, int mods, void *args);

int main(int argc, char **argv)
{

	Timer chrono;

	/* Load Mesh */
	chrono.start();
	Mesh mesh;
	int res = load_mesh(mesh, argc, argv);
	if (res) {
		syntax(argv[0]);
	} else {
		printf("Vertices : %zu | Triangles : %zu\n",
		       mesh.vertex_count(), mesh.triangle_count());
	}
	chrono.stop("loading mesh");

	Viewer viewer;
	init_camera_for_mesh(mesh, viewer.camera);

	viewer.init("Viewer App");
	viewer.register_key_callback({key_cb, &cfg});

	/* Prepare GPU data */
	chrono.start();
	const char *vert_shader = "./shaders/default.vert";
	const char *frag_shader = "./shaders/default.frag";
	int shader = create_shader(vert_shader, frag_shader);
	if (!shader) {
		exit(EXIT_FAILURE);
	}
	GPUMesh gpu_mesh;
	gpu_mesh.m = &mesh;
	gpu_mesh.upload();
	chrono.stop("uploading mesh to GPU");

	while (!viewer.should_close()) {

		viewer.poll_events();
		update_all();
		viewer.begin_frame();
		draw_scene(viewer, shader, gpu_mesh);
		draw_gui();
		viewer.end_frame();
	}

	viewer.fini();

	return (EXIT_SUCCESS);
}

static void syntax(char *prg_name)
{
	printf("Syntax : %s ($(obj_filename)| cube | sphere) [n]", prg_name);
	printf("         Subdivision number n must be provided in case of "
	       "         cube or sphere mesh.\n");
	exit(EXIT_FAILURE);
}

static int load_mesh(Mesh &mesh, int argc, char **argv)
{
	int res = -1;
	if (argc > 2 && strncmp(argv[1], "cube", 4) == 0) {
		res = load_cube(mesh, atoi(argv[2]));
	} else if (argc > 2 && strncmp(argv[1], "sphere", 5) == 0) {
		res = load_sphere(mesh, atoi(argv[2]));
	} else if (argc > 1) {
		res = load_obj(argv[1], mesh);
	}
	return res;
}

static void init_camera_for_mesh(const Mesh &mesh, Camera &camera)
{
	Aabb bbox = compute_mesh_bounds(mesh);
	Vec3 model_center = (bbox.min + bbox.max) * 0.5f;
	Vec3 model_extent = (bbox.max - bbox.min);
	float model_size = max(model_extent);
	if (model_size == 0) {
		printf("Warning : Mesh is empty or reduced to a point.\n");
		model_size = 1;
	}
	camera.set_target(model_center);
	Vec3 start_pos = (model_center + 2.f * Vec3(0, 0, model_size));
	camera.set_position(start_pos);
	camera.set_near(0.01 * model_size);
	camera.set_far(100 * model_size);
}

static void update_all() {}

static void draw_scene(const Viewer &viewer, int shader,
		       const GPUMesh &gpu_mesh)
{
	glClearColor(cfg.bgcolor[0], cfg.bgcolor[1], cfg.bgcolor[2],
		     cfg.bgcolor[3]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	const Camera &camera = viewer.camera;
	glUseProgram(shader);
	Mat4 proj = camera.view_to_clip();
	Mat4 vm = camera.world_to_view();
	Vec3 camera_pos = camera.get_position();
	glUniformMatrix4fv(0, 1, 0, &vm(0, 0));
	glUniformMatrix4fv(1, 1, 0, &proj(0, 0));
	glUniform3fv(2, 1, &camera_pos[0]);

	if (cfg.draw_surface) {
		glEnable(GL_POLYGON_OFFSET_FILL);
		float offset = reversed_z ? -1.f : 1.f;
		glPolygonOffset(offset, offset);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glUniform1i(3, true);
		gpu_mesh.draw();
	}
	if (cfg.draw_edges) {
		glDisable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(0.f, 0.f);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glUniform1i(3, false);
		gpu_mesh.draw();
	}
}

static void draw_gui()
{
	ImGui::Begin("Controls");
	float fps = ImGui::GetIO().Framerate;
	ImGui::Text("Average %.3f ms/frame (%.1f FPS)", 1000.0f / fps, fps);
	ImGui::End();
}

static void key_cb(int key, int action, int mods, void *args)
{
	(void)mods;
	Cfg *cfg = (Cfg *)args;
	if (key == GLFW_KEY_S && action == GLFW_PRESS) {
		cfg->draw_surface = !cfg->draw_surface;
		return;
	}
	if (key == GLFW_KEY_E && action == GLFW_PRESS) {
		cfg->draw_edges = !cfg->draw_edges;
		return;
	}
}

