#pragma once

struct Mouse {
	bool is_pressed[3] = {false};
	bool is_double_click[3] = {false};
	double x;
	double y;
	double last_click_x[3];
	double last_click_y[3];
	double last_click_time[3] = {0};
	int mods[3] = {0};

	void record_button(int button, int action, int mods);
	void record_move(double x, double y);
};
