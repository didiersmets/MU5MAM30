#pragma once

#include "array.h"
#include "my_mesh.h"

struct FEMatrix{
	//enum FEMType {
	//	P1_cst, /* P1 with constant off diag coeffs  */
	//	P1_sym, /* P1 with symmetric off diag coeffs */
	//	P1_gen, /* P1 with general off diag coeffs   */
	//};
	//FEMType fem_type;
	size_t rows;
	size_t cols;
	const Mesh *m;
	TArray<double> diag;
	TArray<double> off_diag;

	void mvp(const double *__restrict x,
		 double *__restrict y) const;
	double sum() const;
};

