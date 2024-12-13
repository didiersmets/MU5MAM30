#pragma once 

#include <iostream>
#include <string.h>
#include <math.h>
#include "tiny_blas.h"
#include "sparse_matrix.h"
#include "fem_matrix.h"
#include "array.h"
#include "sparse_matrix.h"
#include "skyline.h"

int conj_gradient(const struct FEMatrix *S,
			  struct FEMatrix *M, TArray<double> *B,
			  TArray<double> *U,double *error, int N){

    //initial residue

    TArray<double> r(N,0);

	memcpy(r.data, B->data, N * sizeof(double));
	double error2 = blas_dot(r.data,r.data,N);

    TArray<double> Ar(N,0);
	TArray<double> Mr(N,0);

    
    int iterate = 0;
    int iter_max = 1000;
    double tol = 1e-6;
    double tol2 = tol*tol;

	TArray<double> p(N);
	memcpy(p.data, r.data, N * sizeof(double));
	TArray<double> Mp(N);
	TArray<double> Ap(N);

    while(error2 > tol2 && iterate < iter_max){
		M->mvp(p.data,Mp.data);
		S->mvp(p.data,Ap.data);
		blas_axpby(1,Mp.data,1,Ap.data,N);
		double pAp = blas_dot(p.data,Ap.data,N);
		
		double alpha = blas_dot(p.data,r.data,N)/pAp;
		blas_axpby(alpha,p.data,1,U->data,N);
		blas_axpby(-alpha,Ap.data,1,r.data,N);
		S->mvp(r.data,Ar.data);
    	M->mvp(r.data,Mr.data);
    	blas_axpby(1,Mr.data,1,Ar.data,N);
		double beta = blas_dot(p.data,Ar.data,N)/pAp;
		blas_axpby(1,r.data,-beta,p.data,N);
		
	
	//Update error
    error2 = blas_dot(r.data,r.data,N);
    iterate++;
}

*error = sqrt(error2);

return iterate;
}
    

int conj_gradient(const struct CSRMatrix *S,
			  struct CSRMatrix *M, TArray<double> *B,
			  TArray<double> *U,double *error, int N){

    //initial residue

    TArray<double> r(N,0);

	memcpy(r.data, B->data, N * sizeof(double));
	double error2 = blas_dot(r.data,r.data,N);

    TArray<double> Ar(N,0);
	TArray<double> Mr(N,0);

    
    int iterate = 0;
    int iter_max = 1000;
    double tol = 1e-6;
    double tol2 = tol*tol;

	TArray<double> p(N);
	memcpy(p.data, r.data, N * sizeof(double));
	TArray<double> Mp(N);
	TArray<double> Ap(N);

    while(error2 > tol2 && iterate < iter_max){
		M->mvp(p.data,Mp.data);
		S->mvp(p.data,Ap.data);
		blas_axpby(1,Mp.data,1,Ap.data,N);
		double pAp = blas_dot(p.data,Ap.data,N);
		
		double alpha = blas_dot(p.data,r.data,N)/pAp;
		blas_axpby(alpha,p.data,1,U->data,N);
		blas_axpby(-alpha,Ap.data,1,r.data,N);
		S->mvp(r.data,Ar.data);
    	M->mvp(r.data,Mr.data);
    	blas_axpby(1,Mr.data,1,Ar.data,N);
		double beta = blas_dot(p.data,Ar.data,N)/pAp;
		blas_axpby(1,r.data,-beta,p.data,N);
		
	
	//Update error
    error2 = blas_dot(r.data,r.data,N);
    iterate++;
}

*error = sqrt(error2);

return iterate;
}

int conj_gradient(const struct CSRMatrix *M, TArray<double> *B, TArray<double> *U,double *error, int N){

    //initial residue

    TArray<double> r(N,0);

	memcpy(r.data, B->data, N * sizeof(double));
	double error2 = blas_dot(r.data,r.data,N);
	TArray<double> Mr(N,0);

    int iterate = 0;
    int iter_max = 1000;
    double tol = 1e-6;
    double tol2 = tol*tol;

	TArray<double> p(N);
	memcpy(p.data, r.data, N * sizeof(double));
	TArray<double> Mp(N);

    while(error2 > tol2 && iterate < iter_max){
		M->mvp(p.data,Mp.data);
		double pMp = blas_dot(p.data,Mp.data,N);
		
		double alpha = blas_dot(p.data,r.data,N)/pMp;
		blas_axpby(alpha,p.data,1,U->data,N);
		blas_axpby(-alpha,Mp.data,1,r.data,N);
    	M->mvp(r.data,Mr.data);

		double beta = blas_dot(p.data,Mr.data,N)/pMp;
		blas_axpby(1,r.data,-beta,p.data,N);
		
	
	//Update error
    error2 = blas_dot(r.data,r.data,N);
    iterate++;
}

*error = sqrt(error2);

return iterate;
}

int conj_gradient(const struct SkylineMatrix *M, TArray<double> *B, TArray<double> *U,double *error, int N){

    //initial residue

    TArray<double> r(N,0);

	memcpy(r.data, B->data, N * sizeof(double));
	double error2 = blas_dot(r.data,r.data,N);
	TArray<double> Mr(N,0);

    int iterate = 0;
    int iter_max = 1000;
    double tol = 1e-6;
    double tol2 = tol*tol;

	TArray<double> p(N);
	memcpy(p.data, r.data, N * sizeof(double));
	TArray<double> Mp(N);

    while(error2 > tol2 && iterate < iter_max){
		M->mvp(p.data,Mp.data);
		double pMp = blas_dot(p.data,Mp.data,N);
		
		double alpha = blas_dot(p.data,r.data,N)/pMp;
		blas_axpby(alpha,p.data,1,U->data,N);
		blas_axpby(-alpha,Mp.data,1,r.data,N);
    	M->mvp(r.data,Mr.data);

		double beta = blas_dot(p.data,Mr.data,N)/pMp;
		blas_axpby(1,r.data,-beta,p.data,N);
		
	
	//Update error
    error2 = blas_dot(r.data,r.data,N);
    iterate++;
}

*error = sqrt(error2);

return iterate;
}