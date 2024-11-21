#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gl_utils.h"

#include "imgui/imgui.h"

#include "tiny_expr/tinyexpr.h"

#include "cube.h"
#include "logging.h"
#include "mesh.h"
#include "mesh_bounds.h"
#include "mesh_gpu.h"
#include "mesh_io.h"
#include "navier_stokes.h"
#include "ndc.h"
#include "shaders.h"
#include "sphere.h"
#include "viewer.h"

/* Viewer config */
float bgcolor[4] = {0.3, 0.3, 0.3, 1.0};
bool draw_surface = true;
bool draw_edges = false;
float scale_min;
float scale_max;
float mesh_deform = 0;

/* FEM interaction */
bool autoscale = true;
bool started = false;
bool one_step = false;
bool reset = false;

/* Parameters */
float lognu = -4;
float dt = 0.005;
double tol = 1e-6;

/* RHS expression of the PDE */
char rhs_expression[128] =
    "100 * z * exp(-50*z^2) * (1 + 0.5 * cos(20 * theta))";
bool rhs_show_error = false;
double rhs_x, rhs_y, rhs_z, rhs_p, rhs_t, rhs_r;
te_variable rhs_vars[] = {{"x", &rhs_x},   {"y", &rhs_y},     {"z", &rhs_z},
			  {"phi", &rhs_p}, {"theta", &rhs_t}, {"rand", &rhs_r}};
te_expr *te_rhs = NULL;

static void syntax(char *prg_name);
static int load_mesh(Mesh &mesh, int argc, char **argv);
static void rescale_and_recenter_mesh(Mesh &mesh);
static void init_camera_for_mesh(const Mesh &mesh, Camera &camera);
static void update_all(NavierStokesSolver &solver, Mesh &mesh,
		       GPUMesh &mesh_gpu);
static void draw_scene(const Viewer &viewer, int shader,
		       const GPUMesh &gpu_mesh);
static void draw_gui(NavierStokesSolver &solver);
static void key_cb(int key, int action, int mods, void *args);
static void get_attr_bounds(const Mesh &m, float *attr_min, float *attr_max);

void reset_solver(NavierStokesSolver &solver)
{
	for (size_t i = 0; i < solver.N; ++i) {
		rhs_x = solver.m.positions[i].x;
		rhs_y = solver.m.positions[i].y;
		rhs_z = solver.m.positions[i].z;
		rhs_p = atan2(rhs_y, rhs_x);
		rhs_t = atan2(sqrt(rhs_x * rhs_x + rhs_y * rhs_y), rhs_z);
		rhs_r = (double)rand() / RAND_MAX;
		solver.omega[i] = te_eval(te_rhs);
	}

	solver.set_zero_mean(solver.omega.data);
	memset(solver.psi.data, 0, solver.N * sizeof(double));
	solver.t = 0;
}

bool new_rhs(NavierStokesSolver &solver)
{
	srand((int)time(NULL));
	te_expr *test =
	    te_compile(rhs_expression, rhs_vars,
		       sizeof(rhs_vars) / sizeof(rhs_vars[0]), NULL);
	if (!test)
		return false;

	te_free(te_rhs);
	te_rhs = test;

	reset_solver(solver);

	return true;
}

void transfer_to_mesh(const TArray<double> &V, Mesh &m)
{
	m.attr.resize(m.vertex_count());
	for (size_t i = 0; i < m.vertex_count(); ++i) {
		m.attr[i] = V[i];
	}
}

int main(int argc, char **argv)
{
	log_init(0);

	/* Load Mesh */
	Mesh mesh;
	if (load_mesh(mesh, argc, argv)) {
		syntax(argv[0]);
		exit(EXIT_FAILURE);
	}
	LOG_MSG("Loaded mesh.");
	rescale_and_recenter_mesh(mesh);
	LOG_MSG("Mesh rescaled and recentered.");

	/* Prepare FEM data */
	NavierStokesSolver solver(mesh);
	if (!new_rhs(solver)) {
		LOG_MSG("Error loading rhs (expression flawed ?).");
		exit(EXIT_FAILURE);
	}
	transfer_to_mesh(solver.omega, mesh);
	get_attr_bounds(mesh, &scale_min, &scale_max);
	LOG_MSG("Prepared FEM data.");

	/* Get an OpenGL context through a viewer app. */
	Viewer viewer;
	init_camera_for_mesh(mesh, viewer.camera);
	viewer.init("Navier Stokes 2D solver (vorticity formulation)");
	viewer.register_key_callback({key_cb, NULL});
	viewer.mouse.set_double_click_time(-1);
	LOG_MSG("Viewer initialized.");

	/* Prepare GPU data */
	const char *vert_shader = "./shaders/fem.vert";
	const char *frag_shader = "./shaders/fem.frag";
	int shader = create_shader(vert_shader, frag_shader);
	if (!shader) {
		exit(EXIT_FAILURE);
	}
	LOG_MSG("Shader initialized.");
	GPUMesh gpu_mesh;
	gpu_mesh.m = &mesh;
	gpu_mesh.upload();

	/* Main Loop */
	while (!viewer.should_close()) {
		viewer.poll_events();
		update_all(solver, mesh, gpu_mesh);
		viewer.begin_frame();
		draw_scene(viewer, shader, gpu_mesh);
		draw_gui(solver);
		viewer.end_frame();
	}

	viewer.fini();
	log_fini();

	return (EXIT_SUCCESS);
}

static void syntax(char *prg_name)
{
	printf("Syntax : %s ($(obj_filename)| cube | sphere) [n]\n", prg_name);
	printf("         Subdivision number n must be provided in case of "
	       "cube or sphere mesh.\n");
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

static void rescale_and_recenter_mesh(Mesh &mesh)
{
	Aabb bbox = compute_mesh_bounds(mesh);
	Vec3 model_center = (bbox.min + bbox.max) * 0.5f;
	Vec3 model_extent = (bbox.max - bbox.min);
	float model_size = max(model_extent);
	if (model_size == 0) {
		printf("Warning : Mesh is empty or reduced to a point.\n");
		model_size = 1;
	}
	for (size_t i = 0; i < mesh.vertex_count(); ++i) {
		mesh.positions[i] -= model_center;
		mesh.positions[i] /= (model_size / 2);
	}
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

static void get_attr_bounds(const Mesh &m, float *attr_min, float *attr_max)
{
	if (!m.vertex_count())
		return;
	float min = m.attr[0];
	float max = min;
	for (size_t i = 1; i < m.vertex_count(); ++i) {
		if (m.attr[i] < min) {
			min = m.attr[i];
		} else if (m.attr[i] > max) {
			max = m.attr[i];
		}
	}
	*attr_min = min;
	*attr_max = max;
}

static void update_all(NavierStokesSolver &solver, Mesh &mesh,
		       GPUMesh &gpu_mesh)
{
	bool needs_upload = true;
	if (started || one_step) {
		solver.time_step(dt, pow(10, lognu));
		if (one_step) {
			one_step = false;
		}
		transfer_to_mesh(solver.omega, mesh);
		if (autoscale) {
			get_attr_bounds(mesh, &scale_min, &scale_max);
		}
	} else if (reset) {
		reset_solver(solver);
		transfer_to_mesh(solver.omega, mesh);
		get_attr_bounds(mesh, &scale_min, &scale_max);
		reset = false;
	} else {
		needs_upload = false;
	}
	if (needs_upload) {
		gpu_mesh.update_attr();
	}
}

static void draw_scene(const Viewer &viewer, int shader,
		       const GPUMesh &gpu_mesh)
{
	glClearColor(bgcolor[0], bgcolor[1], bgcolor[2], bgcolor[3]);
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
	glUniformMatrix4fv(glGetUniformLocation(shader, "vm"), 1, 0, &vm(0, 0));
	glUniformMatrix4fv(glGetUniformLocation(shader, "proj"), 1, 0, &proj(0, 0));
	glUniform3fv(glGetUniformLocation(shader, "camera_pos"), 1, &camera_pos[0]);
	glUniform1f(glGetUniformLocation(shader, "scale_min"), scale_min);
	glUniform1f(glGetUniformLocation(shader, "scale_max"), scale_max);
	glUniform1f(glGetUniformLocation(shader, "deform"), mesh_deform);

	if (draw_surface) {
		glEnable(GL_POLYGON_OFFSET_FILL);
		float offset = reversed_z ? -1.f : 1.f;
		glPolygonOffset(offset, offset);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glUniform1i(glGetUniformLocation(shader, "lighting"), true);
		gpu_mesh.draw();
	}
	if (draw_edges) {
		glDisable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(0.f, 0.f);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glUniform1i(glGetUniformLocation(shader, "lighting"), false);
		gpu_mesh.draw();
	}
}

static void draw_gui(NavierStokesSolver &solver)
{
	ImGui::Begin("Controls");
	ImGui::Text("Navier Stokes solver");
	ImGui::Text("--------------------");

	ImGui::Text("Enter math expression for initial vorticity below:");
	ImGui::Text("(available variables : x, y, z, phi, theta, rand)");
	ImGui::Text("(zero mean automatically achieved by adding constant)");
	ImGui::InputText("", rhs_expression, IM_ARRAYSIZE(rhs_expression));
	if (ImGui::Button("Apply")) {
		if (!new_rhs(solver)) {
			rhs_show_error = true;
		}
		started = false;
		reset = true;
	}
	if (rhs_show_error) {
		ImGui::Begin("Error");
		ImGui::Text("Syntax error in expresion (missing * ?)");
		if (ImGui::Button("Got it!")) {
			rhs_show_error = false;
		}
		ImGui::End();
	}

	ImGui::Text(" ");
	ImGui::Text("Solution value is represented by color :");
	ImGui::Text("Red = low value, Green = mid, Blue = high.");

	ImGui::Text(" ");

	if (ImGui::Button("Start")) {
		started = true;
	}
	ImGui::SameLine();
	if (ImGui::Button("Stop")) {
		started = false;
	}
	ImGui::SameLine();
	if (ImGui::Button("One step")) {
		if (!started) {
			one_step = true;
		}
	}
	ImGui::SameLine();
	if (ImGui::Button("Reset")) {
		reset = true;
	}
	ImGui::Text("Time : %f", solver.t);
	ImGui::Text("Scale min %.2f Scale max %.2f  (Span : %g)", scale_min,
		    scale_max, scale_max - scale_min);

	ImGui::Text(" ");
	ImGui::Text("Controls");
	ImGui::Text("--------");
	ImGui::Text("Viscosity (negative power of 10):");
	ImGui::SliderFloat("nu", &lognu, -8, 0, "10^(%.1f)");
	ImGui::Text("Time step :");
	ImGui::SliderFloat("dt", &dt, 0.f, 0.01f, "%.4f");
	ImGui::Checkbox("Autoscale colors to bounds", &autoscale);
	ImGui::Checkbox("Show mesh edges", &draw_edges);
	ImGui::Text("Artificially deform mesh according to omega :");
	ImGui::Text("(may help visualize oscillations)");
	ImGui::SliderFloat("  ", &mesh_deform, 0.f, 1.f);

	ImGui::Text(" ");
	ImGui::Text("Number of DOF : %zu", solver.N);
	float fps = ImGui::GetIO().Framerate;
	ImGui::Text("Average framerate : %.1f FPS", fps);

	ImGui::Text(" ");
	ImGui::Text("Mouse :");
	ImGui::Text("Click + drag : orbit");
	ImGui::Text("Click + CTRL + drag : zoom in/out");
	ImGui::Text("Click + SHIFT + drag : translate");

	ImGui::End();
}

static void key_cb(int key, int action, int mods, void *args)
{
	(void)mods;
	(void)args;
	if (key == GLFW_KEY_S && action == GLFW_PRESS) {
		draw_surface = !draw_surface;
		return;
	}
	if (key == GLFW_KEY_E && action == GLFW_PRESS) {
		draw_edges = !draw_edges;
		return;
	}
}

