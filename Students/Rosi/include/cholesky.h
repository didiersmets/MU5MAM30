#pragma once

#include "skyline.h"
#include "math.h"
#include "array.h"

void Cholesky(const struct SkylineMatrix *S, struct SkylineMatrix *L){
    int N = S->rows;
    int count = 0;
    
    L->cols = S->cols;
    L->nnz = S->nnz;
    L->rows = S->rows;
    L->symmetric = false;
    L->start.resize(N);
    L->J.resize(N);
    L->val.resize(L->nnz);
    
    for(size_t i = 0; i < L->nnz; i++)
        L->val[i] = 0.0;
    for(size_t i = 0; i < N; i++){
        L->start[i] = S->start[i];
        L->J[i] = S->J[i];
    }
    
    for (int i = 0; i < N; i++){
        int pointer = S->J[i];
        L->val[L->start[i] + i - pointer] = S->val[S->start[i] + i - pointer];
        for (int j = pointer; j < i; j++){
            
            L->val[S->start[i] + j - pointer ] = S->val[S->start[i] + j - pointer];
            for (int k = S->J[i]; k < j; k++){
                L->val[S->start[i] + j - pointer] -= L->val[S->start[i] + j - k]*L->val[S->start[j] + j - k];
            }
            L->val[S->start[i] + j - pointer] /= L->val[L->start[j + 1] - 1];
            L->val[L->start[i] + i - pointer] -= L->val[S->start[i] + j - pointer]*L->val[S->start[i] + j - pointer];
        }
        L->val[L->start[i] + i - pointer] = sqrt(L->val[L->start[i] + i - pointer]);
    }
}

// Solve Ax = b with A = LL^T

void CholeskySolve(const struct SkylineMatrix *L, TArray<double> *b, TArray<double> *x){
    int N = L->rows;
    TArray<double> y(N,0);
    TArray<double> z(N,0);
    for (int i = 0; i < N; i++){
        y[i] = (*b)[i];
        for (int j = L->J[i]; j < i; j++){
            y[i] -= L->val[L->start[i] + i - L->J[i]]*y[j];
        }
        y[i] /= L->val[L->start[i] + i - L->J[i]];
    }
    for (int i = N-1; i >= 0; i--){
        z[i] = y[i];
        for (int j = i+1; j < N; j++){
            z[i] -= L->val[L->start[j] + i - L->J[j]]*z[j];
        }
        assert(L->val[L->start[i] + i - L->J[i]] != 0);
        z[i] /= L->val[L->start[i] + i - L->J[i]];
    }
    for (int i = 0; i < N; i++){
        (*x)[i] = z[i];
    }
}