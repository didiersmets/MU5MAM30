#include <assert.h>

#include <GLFW/glfw3.h>

#include "mouse.h"

#define DOUBLE_CLICK_TIME 0.5 /* in seconds           */

void Mouse::record_button(int button, int action, int mods)
{
	if (action == GLFW_PRESS) {
		assert(button >= 0 && button < 3);
		is_pressed[button] = true;
		double now = glfwGetTime();
		is_double_click[button] =
		    (now - last_click_time[button]) < DOUBLE_CLICK_TIME;
		last_click_x[button] = x;
		last_click_y[button] = y;
		last_click_time[button] = now;
		this->mods[button] = mods;
	} else if (action == GLFW_RELEASE) {
		assert(button >= 0 && button < 3);
		is_pressed[button] = false;
	}
}

void Mouse::record_move(double x, double y)
{
	this->x = x;
	this->y = y;
}
