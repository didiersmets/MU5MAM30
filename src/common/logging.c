#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "logging.h"
#include "sys_utils.h"

#define DATE_FMT "%Y-%m-%d %H:%M:%S"
#define PREFIX_FMT "%s [%s:%d]: ", timedate, filename, line

static void (*logfunc)(const char *) = NULL;

void logprint(const char *buf) { printf("%s", buf); }

void log_init(void (*func)(const char *)) { logfunc = func ? func : logprint; }

void log_fini() { logfunc = NULL; }

void log_impl(const char *filename, int line, const char *fmt, ...)
{
	if (!logfunc)
		return;
	va_list args;
	va_start(args, fmt);
	vlog_impl(filename, line, fmt, args);
	va_end(args);
}

void vlog_impl(const char *filename, int line, const char *fmt, va_list args)
{
	if (!logfunc)
		return;
	time_t t = time(NULL);
	struct tm *now = localtime(&t);
	char timedate[24];
	if (!strftime(timedate, sizeof(timedate), DATE_FMT, now)) {
		logfunc("Internal error. Cannot log timedate.\n");
		abort();
	}

	size_t prefix_len = snprintf(NULL, 0, PREFIX_FMT);
	va_list args_copy;
	va_copy(args_copy, args);
	size_t len = vsnprintf(NULL, 0, fmt, args_copy);

	char *buf = (char *)malloc(prefix_len + len + 2);
	if (!buf) {
		logfunc("Internal error. Out of memory while logging.\n");
		abort();
	}

	(void)snprintf(buf, prefix_len + 1, PREFIX_FMT);
	(void)vsnprintf(&buf[prefix_len], len + 1, fmt, args);
	(void)sprintf(&buf[strlen(buf)], "\n");
	logfunc(buf);
	free(buf);
}
