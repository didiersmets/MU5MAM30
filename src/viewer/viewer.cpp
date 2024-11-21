#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gl_utils.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include "viewer.h"

#include "camera.h"
#include "mouse.h"
#include "ndc.h"
#include "trackball.h"

#define ZOOM_SENSITIVITY 0.3f /* for mouse zoom */

#ifndef __APPLE__
static void GL_debug_cb(GLenum source, GLenum type, GLuint id, GLenum severity,
			GLsizei length, const GLchar *message,
			const void *user_param);
#endif
static void mouse_button_cb(GLFWwindow *window, int button, int action,
			    int mods);
static void cursor_pos_cb(GLFWwindow *window, double x, double y);
static void scroll_cb(GLFWwindow *window, double xoffset, double yoffset);
static void key_cb(GLFWwindow *window, int key, int scancode, int action,
		   int mods);
static void resize_window_cb(GLFWwindow *window, int width, int height);

void Viewer::init(const char *name)
{
	this->name = name;

	/* Set-up GLFW and */
	if (!glfwInit()) {
		printf("Cannot initialize GLFW.\n");
		exit(EXIT_FAILURE);
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
	glfwWindowHint(GLFW_DEPTH_BITS, 32);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_SAMPLES, 4);
	window = glfwCreateWindow(1920, 1080, name, NULL, NULL);
	if (!window) {
		printf("Cannot create GLFW window.\n");
		exit(EXIT_FAILURE);
	}
	glfwGetWindowSize(window, &width, &height);
	camera.set_aspect((float)width / height);
	glfwSetWindowUserPointer(window, this);
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, key_cb);
	glfwSetFramebufferSizeCallback(window, resize_window_cb);
	glfwSetCursorPosCallback(window, cursor_pos_cb);
	glfwSetMouseButtonCallback(window, mouse_button_cb);
	glfwSetScrollCallback(window, scroll_cb);
	glfwSwapInterval(1);

	/* Allow OpenGL debug messages */
#ifndef __APPLE__
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(GL_debug_cb, NULL);
#endif
	/* Set-up OpenGL for our choice of NDC */
	set_up_opengl_for_ndc();


	/* Set-up Imgui */
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 150");
}

void Viewer::register_key_callback(KeyCallback cb) { key_callback = cb; }

bool Viewer::should_close() const { return glfwWindowShouldClose(window); }

void Viewer::poll_events() { return glfwPollEvents(); }

void Viewer::begin_frame()
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

void Viewer::end_frame()
{
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glfwSwapBuffers(window);
}

void Viewer::fini()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	window = NULL;
	glfwTerminate();
}

#ifndef __APPLE__
static void GL_debug_cb(GLenum source, GLenum type, GLuint id, GLenum severity,
			GLsizei length, const GLchar *message,
			const void *user_param)
{
	(void)source;
	(void)length;
	(void)user_param;
	(void)id;
	if (type == GL_DEBUG_TYPE_ERROR) {
		printf("GL CALLBACK: %s type = 0x%x, severity = 0x%x,\
			message = %s\n",
		       type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "",
		       type, severity, message);
	}
}
#endif

static void mouse_button_cb(GLFWwindow *window, int button, int action,
			    int mods)
{
	if (ImGui::GetIO().WantCaptureMouse) {
		return;
	}

	Viewer *viewer = (Viewer *)glfwGetWindowUserPointer(window);
	Camera &camera = viewer->camera;
	Mouse &mouse = viewer->mouse;
	ScreenTrackball &trackball = viewer->trackball;
	int width = viewer->width;
	int height = viewer->height;

	viewer->mouse.record_button(button, action, mods);

	if (button == 0 && action == GLFW_PRESS) {
		float px = mouse.x;
		float py = mouse.y;
		trackball.grab(px, py, width, height);
		camera.save_spatial_state();
		if (!mouse.is_double_click[button])
			return;
		float depth;
		float tx = px;
		float ty = height - py;
		glReadPixels(tx, ty, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT,
			     &depth);
		/* Don't track clicks outisde of model */
		if (approx_equal(depth, 1.f - reversed_z)) {
			return;
		}
		Vec3 target =
		    camera.world_coord_at(px / width, py / height, depth);
		camera.set_target(target);
	}
}

static void cursor_pos_cb(GLFWwindow *window, double x, double y)
{
	Viewer *viewer = (Viewer *)glfwGetWindowUserPointer(window);
	Camera &camera = viewer->camera;
	Mouse &mouse = viewer->mouse;
	ScreenTrackball &trackball = viewer->trackball;
	int width = viewer->width;
	int height = viewer->height;

	viewer->mouse.record_move(x, y);

	if (!mouse.is_pressed[0])
		return;

	int mods = mouse.mods[0];
	float delta_x = x - mouse.last_click_x[0];
	float delta_y = y - mouse.last_click_y[0];

	if (mods & GLFW_MOD_SHIFT) {
		/* Translation in x and y */
		float dist = norm(camera.get_target() - camera.get_position());
		float mult = dist / width;
		Vec3 trans = {-delta_x * mult, delta_y * mult, 0.f};
		camera.restore_spatial_state();
		camera.translate(trans, Camera::View);
	} else if (mods & GLFW_MOD_CONTROL) {
		/* Zoom (translation in target direction) */
		float factor = exp(ZOOM_SENSITIVITY * delta_x / 100);
		camera.restore_spatial_state();
		camera.zoom(factor);
	} else /* no modifier */
	{
		/* Orbit around target */
		bool needs_reset = false;
		Quat rot = trackball.drag(x, y, width, height, &needs_reset);
		camera.restore_spatial_state();
		camera.orbit(-rot);
		if (needs_reset) {
			camera.save_spatial_state();
		}
	}
}

static void scroll_cb(GLFWwindow *window, double xoffset, double yoffset)
{
	if (ImGui::GetIO().WantCaptureMouse) {
		return;
	}

	(void)xoffset;
	Viewer *viewer = (Viewer *)glfwGetWindowUserPointer(window);
	Camera &camera = viewer->camera;
	/* Zoom (translation in target direction) */
	float factor = exp(ZOOM_SENSITIVITY * yoffset);
	camera.zoom(factor);
}

static void key_cb(GLFWwindow *window, int key, int scancode, int action,
		   int mods)
{
	(void)scancode;

	if (ImGui::GetIO().WantCaptureKeyboard) {
		return;
	}

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, 1);
		return;
	}

	/* Transmit callback to calling app */
	Viewer *viewer = (Viewer *)glfwGetWindowUserPointer(window);
	const KeyCallback &cb = viewer->key_callback;
	if (cb.func) {
		cb.func(key, action, mods, cb.args);
	}
}

static void resize_window_cb(GLFWwindow *window, int width, int height)
{
	Viewer *viewer = (Viewer *)glfwGetWindowUserPointer(window);

	glfwGetWindowSize(window, &viewer->width, &viewer->height);
	viewer->camera.set_aspect((float)width / height);
	glViewport(0, 0, width, height);
}

