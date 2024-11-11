#pragma once

struct Mouse {
	bool is_pressed[3] = {false, false, false};
	bool is_double_click[3] = {false, false, false};
	double x;
	double y;
	double last_click_x[3];
	double last_click_y[3];
	double last_click_time[3] = {0, 0, 0};
	int mods[3] = {0};
	double double_click_time = 0.5;

	void set_double_click_time(double val);
	void record_button(int button, int action, int mods);
	void record_move(double x, double y);
};
