#pragma once
#include <GLFW/glfw3.h>

#include "camera.h"
#include "mouse.h"
#include "trackball.h"

struct KeyCallback {
	void (*func)(int key, int action, int mods, void *args) = NULL;
	void *args = NULL;
};

struct Viewer {
	const char *name;
	GLFWwindow *window;
	int width;
	int height;
	Camera camera;
	Mouse mouse;
	ScreenTrackball trackball;
	KeyCallback key_callback;

	/* Methods */
	void init(const char *name);
	void register_key_callback(KeyCallback cb);
	bool should_close() const;
	void poll_events();
	void begin_frame();
	void end_frame();
	void fini();
};

