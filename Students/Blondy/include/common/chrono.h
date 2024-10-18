#pragma once
/******************************************************************************
 * Timer : A basic chrono to profile your code. 
 *****************************************************************************/

#include <sys/time.h>

struct Timer {
	bool launched = false;
	timeval start_time;
	void start();
	unsigned int stop(const char *str);
};
