#pragma once

#include "array.h"
#include "matrix.h"
#include "mesh.h"

struct FEMatrix : public Matrix {
	enum FEMType {
		P1_cst, /* P1 with constant off diag coeffs  */
		P1_sym, /* P1 with symmetric off diag coeffs */
		P1_gen, /* P1 with general off diag coeffs   */
		/* TODO Deal with more FEM types             */
	};
	FEMType fem_type;
	const Mesh *m;
	TArray<double> diag;
	TArray<double> off_diag;

	void mvp(const double *__restrict x,
		 double *__restrict y) const override final;
};

