#pragma once

#include <stddef.h>

/* An abstract matrix class */
struct Matrix {
	size_t rows;
	size_t cols;
	/* Matrix vector product : Ax -> y */
	virtual void mvp(const double *__restrict x,
			 double *__restrict y) const = 0;
	/* Sum of matrix elements */
	virtual double sum() const = 0;
};
