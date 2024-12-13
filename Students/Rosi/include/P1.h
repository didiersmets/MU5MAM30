#pragma once

#include "fem_matrix.h"
#include "sparse_matrix.h"
#include "skyline.h"

void mvp_P1(const FEMatrix &A, const double *x, double *y);
double sum_P1(const FEMatrix &A);

void build_P1_CSRPattern(const Mesh &m, CSRPattern &P);

void build_P1_mass_matrix(const Mesh &m, FEMatrix &M);
void build_P1_mass_matrix(const Mesh &m, const CSRPattern &P, CSRMatrix &M);
void build_P1_stiffness_matrix(const Mesh &m, FEMatrix &S);
void build_P1_stiffness_matrix(const Mesh &m, const CSRPattern &P,
			       CSRMatrix &S);
void build_P1_stiffness_matrix(const Mesh &m, SkylineMatrix &S);