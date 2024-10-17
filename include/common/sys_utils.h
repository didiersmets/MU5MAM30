#pragma once
/******************************************************************************
 * Sys Utils : A few macros to take advantage of compiler optimizations, 
 *             and "safe" (i.e. exit if fail...) memory allocation versions
 *             of malloc / calloc / realloc.
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) (__builtin_expect((x), 1))
#define UNLIKELY(x) (__builtin_expect((x), 0))
#define VOIDSTAR_TO_TYPE(x) (typeof(x))
#define MAYBE_UNUSED [[maybe_unused]]
#define ASSUME(cond)                             \
	do {                                     \
		if (!(cond))                     \
			__builtin_unreachable(); \
	} while (0)
#define ASSUME_ALIGNED(var, size)                            \
	do {                                                 \
		var = VOIDSTAR_TO_TYPE(var)                  \
			__builtin_assume_aligned(var, size); \
	} while (0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define ASSUME(cond)
#define ASSUME_ALIGNED(var, n)
#define MAYBE_UNUSED
#endif

#define ASSERT_ALWAYS(x)                                                     \
	do {                                                                 \
		if (UNLIKELY(!(x))) {                                        \
			printf("ERROR: Condition (%s) failed (%s:%d)\n", #x, \
			       __FILE__, __LINE__);                          \
			abort();                                             \
		}                                                            \
	} while (0)

#ifndef NDEBUG
#define ASSERT(x) ASSERT_ALWAYS(x)
#else
#define ASSERT(x) (void)(x)
#endif

MAYBE_UNUSED static inline void *safe_malloc(size_t size)
{
	void *p = malloc(size);
	if (UNLIKELY(!p && size > 0)) {
		abort();
	}
	return (p);
}

MAYBE_UNUSED static inline void *safe_calloc(size_t num, size_t size)
{
	void *p = calloc(num, size);
	if (UNLIKELY(!p && num > 0 && size > 0)) {
		abort();
	}
	return (p);
}

MAYBE_UNUSED static inline void *safe_realloc(void *ptr, size_t size)
{
	void *p = realloc(ptr, size);
	if (UNLIKELY(!p && size > 0)) {
		abort();
	}
	return (p);
}

// Remove old 16bit Windows.h macros near and far
#ifdef far
#undef far
#endif
#ifdef near
#undef near
#endif
