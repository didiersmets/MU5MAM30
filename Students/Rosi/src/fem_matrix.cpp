#include "fem_matrix.h"

#include "P1.h"

void FEMatrix::mvp(const double *x, double *y) const
{
	mvp_P1(*this, x, y);
}

double FEMatrix::sum() const
{
	return sum_P1(*this);
}
