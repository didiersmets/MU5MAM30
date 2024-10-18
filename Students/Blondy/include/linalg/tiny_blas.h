#pragma once
#include <stddef.h>
#include <stdio.h>
#include <string.h>

void inline blas_copy(const double *src, double *dest, size_t N)
{
	memcpy(dest, src, N * sizeof(double));
}

void inline blas_axpy(double a, const double *__restrict x,
		      double *__restrict y, size_t N)
{
	for (size_t i = 0; i < N; ++i) {
		y[i] += a * x[i];
	}
}

void inline blas_axpby(double a, const double *__restrict x, double b,
		       double *__restrict y, size_t N)
{
	for (size_t i = 0; i < N; ++i) {
		y[i] = a * x[i] + b * y[i];
	}
}

double inline blas_dot(const double *x, const double *y, size_t N)
{
	double res = 0.0;
	for (size_t i = 0; i < N; ++i) {
		res += x[i] * y[i];
	}
	return (res);
}
