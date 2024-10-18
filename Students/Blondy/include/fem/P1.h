#pragma once

#include "fem_matrix.h"

void mvp_P1_cst(const FEMatrix &A, const double *x, double *y);
void mvp_P1_sym(const FEMatrix &A, const double *x, double *y);
void mvp_P1_gen(const FEMatrix &A, const double *x, double *y);

void build_P1_mass_matrix(const Mesh &m, FEMatrix &S);
void build_P1_stiffness_matrix(const Mesh &m, FEMatrix &S);
