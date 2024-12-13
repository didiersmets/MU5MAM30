#include "skyline.h"
#include "math_utils.h"


double &SkylineMatrix::operator()(uint32_t i, uint32_t j){
    assert(i < rows);
    assert(j <= i);
    assert(J[i] <= j);
    return val[start[i] + i - j];
}

const double& SkylineMatrix::operator()(uint32_t i, uint32_t j) const {
    assert(i < rows);
    assert(j <= i);
    assert(J[i] <= j);
    return val[start[i] + i - j];
}

void SkylineMatrix::mvp(const double *__restrict x, double *__restrict y) const{
    for (size_t i = 0; i < rows; i++){
        y[i] = 0;
    }
    for(size_t i = 0; i < rows; i++){
        for(size_t j = J[i]; j < i + 1; j++){
            y[i] += this->operator()(i,j) * x[j];
            if(symmetric && i != j)
                y[j] += this->operator()(i,j)*x[i];
        }
    }
}

double SkylineMatrix::sum() const{
    double res = 0.0;
    for(size_t i = 0; i < nnz; i++){
        res += val[i]; 
    }
    if(symmetric){
        res *= 2;
        for(int i = 0; i < rows; i++){
            res -= val[start[i]];
        }
    }
    return res;
}