#include "fem_matrix.h"

#include "P1.h"

void FEMatrix::mvp(const double *x, double *y) const
{
	switch (fem_type) {
	case FEMatrix::P1_cst:
		mvp_P1_cst(*this, x, y);
		return;
	case FEMatrix::P1_sym:
		mvp_P1_sym(*this, x, y);
		return;
	case FEMatrix::P1_gen:
		mvp_P1_gen(*this, x, y);
		return;
	}
}

double FEMatrix::sum() const
{
	switch (fem_type) {
	case FEMatrix::P1_cst:
		return sum_P1_cst(*this);
	case FEMatrix::P1_sym:
		return sum_P1_sym(*this);
	case FEMatrix::P1_gen:
		return sum_P1_gen(*this);
	default:
		return 0;
	}
}
