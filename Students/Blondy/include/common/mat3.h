#pragma once
/******************************************************************************
 * Mat3 : 3x3 matrices
 *        Note : in column major memory order, but the (,) operator is 
 *               overloaded so that M(i,j) yields the coefficient of M at 
 *               line index i and colum index j (0 \leq i,j \leq 2)
 *****************************************************************************/
#include <assert.h>
#include <stdio.h>

#include "vec3.h"

template <typename T> struct TMat3 {
	TVec3<T> cols[3];

	/* Element accessor and mutator */
	const T &operator()(int i, int j) const;
	T &operator()(int i, int j);

	/* Column acessor and mutator */
	const TVec3<T> &operator()(int i) const;
	TVec3<T> &operator()(int i);
};

typedef TMat3<float> Mat3f;
typedef TMat3<double> Mat3d;
using Mat3 = Mat3f;

/* Free functions */

template <typename T>
TMat3<T> operator*(const TMat3<T> &m1, const TMat3<T> &m2);

/* Implementations */

template <typename T> inline const T &TMat3<T>::operator()(int i, int j) const
{
	assert(i >= 0 && i <= 2 && j >= 0 && j <= 2);

	return cols[j][i];
}

template <typename T> inline T &TMat3<T>::operator()(int i, int j)
{
	assert(i >= 0 && i <= 2 && j >= 0 && j <= 2);

	return cols[j][i];
}

template <typename T> inline const TVec3<T> &TMat3<T>::operator()(int j) const
{
	assert(j >= 0 && j <= 2);

	return cols[j];
}

template <typename T> inline TVec3<T> &TMat3<T>::operator()(int j)
{
	assert(j >= 0 && j <= 2);

	return cols[j];
}

template <typename T> TMat3<T> operator*(const TMat3<T> &A, const TMat3<T> &B)
{
	TMat3<T> AB;

	AB(0) = A(0) * B(0, 0) + A(1) * B(1, 0) + A(2) * B(2, 0);
	AB(1) = A(0) * B(0, 1) + A(1) * B(1, 1) + A(2) * B(2, 1);
	AB(2) = A(0) * B(0, 2) + A(1) * B(1, 2) + A(2) * B(2, 2);
	AB(3) = A(0) * B(0, 3) + A(1) * B(1, 3) + A(2) * B(2, 3);

	return (AB);
}

template <typename T> void print(const TMat3<T> &m)
{
	printf("%.3f %.3f %.3f\n", m(0, 0), m(0, 1), m(0, 2));
	printf("%.3f %.3f %.3f\n", m(1, 0), m(1, 1), m(1, 2));
	printf("%.3f %.3f %.3f\n", m(2, 0), m(2, 1), m(2, 2));
}
