#include <stdio.h>
#include <sys/time.h>

#include "chrono.h"

void Timer::start()
{
	gettimeofday(&start_time, NULL);
	launched = true;
}

unsigned int Timer::stop(const char *str)
{
	if (!launched)
		return 0;
	timeval stop_time;
	gettimeofday(&stop_time, NULL);
	unsigned int mus = 1000000 * (stop_time.tv_sec - start_time.tv_sec);
	mus += (stop_time.tv_usec - start_time.tv_usec);
	if (str[0]) {
		printf("Timer %s: ", str);
		if (mus >= 1000000) {
			printf("%.3f s\n", (float)mus / 1000000);
		} else {
			printf("%.3f ms\n", (float)mus / 1000);
		}
	}
	launched = false;
	return (mus);
}

