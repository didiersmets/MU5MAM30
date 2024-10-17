#include <stdio.h>
#include <stdlib.h>

#ifndef GL_GLEXT_PROTOTYPES
	#define GL_GLEXT_PROTOTYPES 1
#endif
#include <GL/gl.h>
#include <GL/glext.h>
#include <GLFW/glfw3.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include "gui_app.h"
#include "ndc.h"

GUIApp::GUIApp(const char *name)
{
	this->name = name;
	window = NULL;
	width = 1920;
	height = 1080;
	bgcolor[0] = .75f;
	bgcolor[1] = .85f;
	bgcolor[2] = .95f;
	bgcolor[3] = 1.f;
}

// GUIApp::~GUIApp() {}

void GUIApp::start()
{
	create_window();
	main_loop();
	destroy_window();
}

void GUIApp::stop() { glfwSetWindowShouldClose(window, 1); }

void GUIApp::mouse_button_callback(int button, int action, int mod)
{
	(void)mod;
	mouse.button(button, action, mod);
}

void GUIApp::cursor_pos_callback(double xpos, double ypos)
{
	mouse.move(xpos, ypos);
}

void GUIApp::scroll_callback(double xoffset, double yoffset)
{
	(void)xoffset;
	(void)yoffset;
}

void GUIApp::key_callback(int key, int scancode, int action, int mods)
{
	(void)key;
	(void)scancode;
	(void)action;
	(void)mods;
}

void GUIApp::create_window()
{
	/* Set-up GLFW */
	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
	glfwWindowHint(GLFW_DEPTH_BITS, 32);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_SAMPLES, 4);

	window = glfwCreateWindow(width, height, name, NULL, NULL);

	if (!window) {
		exit(EXIT_FAILURE);
	}
	glfwSetWindowUserPointer(window, this);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
}

void GUIApp::main_loop()
{
	bool initialized = false;

	while (!should_close()) {
		if (!initialized) {
			opengl_init();
			imgui_init();
			callbacks_init();
			initialized = true;
		}
		one_frame();
	}

	if (initialized) {
		imgui_fini();
		opengl_fini();
	}
}

bool GUIApp::should_close() { return glfwWindowShouldClose(window); }

static void GL_debug_callback(GLenum source, GLenum type, GLuint id,
			      GLenum severity, GLsizei length,
			      const GLchar *message, const void *user_param)
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
void GUIApp::opengl_init()
{
	/* Allow OpenGL debug messages */
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(GL_debug_callback, NULL);

	/* Set-up OpenGL for our choice of NDC */
	set_up_opengl_for_ndc();
}

void GUIApp::imgui_init()
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 150");
}

static void mouse_button_callback(GLFWwindow *window, int button, int action,
				  int mods)
{
	if (ImGui::GetIO().WantCaptureMouse) {
		return;
	}
	GUIApp *app = static_cast<GUIApp *>(glfwGetWindowUserPointer(window));
	app->mouse_button_callback(button, action, mods);
}

static void cursor_pos_callback(GLFWwindow *window, double x, double y)
{
	GUIApp *app = static_cast<GUIApp *>(glfwGetWindowUserPointer(window));
	app->cursor_pos_callback(x, y);
}

static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
	if (ImGui::GetIO().WantCaptureMouse) {
		return;
	}
	GUIApp *app = static_cast<GUIApp *>(glfwGetWindowUserPointer(window));
	app->scroll_callback(xoffset, yoffset);
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action,
			 int mods)
{
	if (ImGui::GetIO().WantCaptureKeyboard) {
		return;
	}
	GUIApp *app = static_cast<GUIApp *>(glfwGetWindowUserPointer(window));
	app->key_callback(key, scancode, action, mods);
}

void GUIApp::callbacks_init()
{
	glfwSetMouseButtonCallback(window, ::mouse_button_callback);
	glfwSetCursorPosCallback(window, ::cursor_pos_callback);
	glfwSetScrollCallback(window, ::scroll_callback);
	glfwSetKeyCallback(window, ::key_callback);
}

void GUIApp::one_frame()
{
	/* Resize if needed */
	int cur_width;
	int cur_height;
	glfwGetWindowSize(window, &cur_width, &cur_height);
	if (cur_width != width || cur_height != height) {
		resize(cur_width, cur_height);
	}

	/* Keys and Mouse callbacks */
	glfwPollEvents();

	/* Prepare states and buffers for drawing */
	pre_draw();

	/* Draw background scene */
	draw_scene();

	/* Draw GUI */
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	draw_gui();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	/* Send frame to screen */
	glfwSwapBuffers(window);
}

void GUIApp::resize(int new_width, int new_height)
{
	width = new_width;
	height = new_height;
}

void GUIApp::pre_draw()
{
	glClearColor(bgcolor[0], bgcolor[1], bgcolor[2], bgcolor[3]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	glDisable(GL_CULL_FACE);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glViewport(0, 0, width, height);
}

void GUIApp::draw_scene() {}

void GUIApp::post_draw() {}

void GUIApp::draw_gui() {}

void GUIApp::imgui_fini()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void GUIApp::opengl_fini() {}

void GUIApp::destroy_window()
{
	glfwDestroyWindow(window);

	window = NULL;
	glfwTerminate();
}

