#pragma once

#include "mouse.h"
#include <GLFW/glfw3.h>

struct GUIApp {
	/* Methods */
      public:
	GUIApp(const char *name);
	void start();
	void stop();
	virtual void mouse_button_callback(int button, int action, int mods);
	virtual void cursor_pos_callback(double x, double y);
	virtual void scroll_callback(double xoffset, double yoffset);
	virtual void key_callback(int key, int scancode, int action, int mods);

      protected:
	void create_window();
	void main_loop();
	bool should_close();
	virtual void opengl_init();
	void imgui_init();
	void callbacks_init();
	void one_frame();
	virtual void resize(int width, int height);
	virtual void pre_draw();
	virtual void draw_scene();
	virtual void post_draw();
	virtual void draw_gui();
	void imgui_fini();
	virtual void opengl_fini();
	void destroy_window();

	/* Members */
      protected:
	const char *name;
	GLFWwindow *window;
	int width;
	int height;
	float bgcolor[4];
	struct Mouse mouse;
};

