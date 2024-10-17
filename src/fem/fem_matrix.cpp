#include <string.h>

#include "fem_matrix.h"
#include "P1.h"

void FEMatrix::mvp(const double *x, double *y) const
{
	switch (fem_type) {
	case FEMatrix::P1_cst:
		mvp_P1_cst(*this, x, y);
		break;
	case FEMatrix::P1_sym:
		mvp_P1_sym(*this, x, y);
		break;
	case FEMatrix::P1_gen:
		mvp_P1_gen(*this, x, y);
		break;
	}
}
