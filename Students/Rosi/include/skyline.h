#pragma once
#include <stdint.h>

#include "array.h"


struct SkylineMatrix {
	size_t rows;
    size_t cols;
    size_t nnz;
    bool symmetric = false;
    TArray<int> J;
    TArray<double> val;
    TArray<int> start;
	void mvp(const double *__restrict x, double *__restrict y) const;
	double sum() const;
	double &operator()(uint32_t i, uint32_t j);
    const double &operator()(uint32_t i, uint32_t j) const;
};