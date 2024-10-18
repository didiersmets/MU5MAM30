#pragma once
/******************************************************************************
 * Logging messages with time stamp, and file and line call site. 
 * Use the utility Macros, not the _impl functions, so that the C-preprocessor 
 * will fill the correct file (base)name and line nbr automatically.
 * Initialization : log_init with user defined logging functions (defaults to
 *                  printf if you pass NULL)
 * Usage          : Similar to printf, i.e. LOG_MSG(const char *fmt, ...)
 * *****************************************************************************/

#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

void log_init(void (*logfunc)(const char *));
void log_fini();
void log_impl(const char *filename, int line, const char *fmt, ...);
void vlog_impl(const char *filename, int line, const char *fmt, va_list args);

/* Macro helpers */

#if defined(__GNUC__) || defined(__clang__)
#define BUILTIN_STRRCHR __builtin_strrchr
#else
#define BUILTIN_STRRCHR strrchr
#endif

#define LOG_BASENAME(f)                                                     \
	(BUILTIN_STRRCHR(f, '/') ?                                          \
		 BUILTIN_STRRCHR(f, '/') + 1 :                              \
		 (BUILTIN_STRRCHR(f, '\\') ? BUILTIN_STRRCHR(f, '\\') + 1 : \
					     (f)))

#define LOG_MSG(...) log_impl(LOG_BASENAME(__FILE__), __LINE__, __VA_ARGS__)

#ifdef ASSERT_ALWAYS
#undef ASSERT_ALWAYS
#undef ASSERT
#define ASSERT_ALWAYS(x)                                           \
	do {                                                       \
		if (UNLIKELY(!(x))) {                              \
			log_impl(LOG_BASENAME(__FILE__), __LINE__, \
				 "Assertion \"%s\" failed.", #x);  \
			abort();                                   \
		}                                                  \
	} while (0)
#ifndef NDEBUG
#define ASSERT(x) ASSERT_ALWAYS(x)
#else
#define ASSERT(x) (void)(x)
#endif
#endif

#ifdef __cplusplus
}
#endif
