#pragma once

/******************************************************************************
 * TArray : Dynamically sized arrays. 
 *          T stand for Templated. 
 *          Template parameter T must be trivially (i.e. bitwise) copiable.
 *****************************************************************************/

#include <assert.h>

#include "sys_utils.h"

template <typename T> struct TArray {
	/**
	 * Members
	 */
	size_t size;
	size_t capacity;
	T *data;
	/**
	 * Methods
	 */
	TArray();
	TArray(size_t size);
	TArray(size_t size, T val);
	TArray(const TArray<T> &other) = delete;
	TArray &operator=(const TArray<T> &other) = delete;
	~TArray();
	T &operator[](size_t i);
	const T &operator[](size_t i) const;
	void push_back(const T &t);
	void resize(size_t size);
	void reserve(size_t capacity);
	void clear();
};

template <typename T>
TArray<T>::TArray()
	: size{ 0 }
	, capacity{ 0 }
	, data{ nullptr }
{
}

template <typename T>
TArray<T>::TArray(size_t size)
	: size{ size }
	, capacity{ size }
{
	data = static_cast<T *>(safe_malloc(size * sizeof(T)));
};

template <typename T>
TArray<T>::TArray(size_t size, T val)
	: size{ size }
	, capacity{ size }
{
	data = static_cast<T *>(safe_malloc(size * sizeof(T)));
	for (size_t i = 0; i < size; ++i) {
		data[i] = val;
	}
};

template <typename T> inline TArray<T>::~TArray()
{
	size = 0;
	capacity = 0;
	free(data);
	data = nullptr;
}

template <typename T> inline T &TArray<T>::operator[](size_t i)
{
	assert(i < size);
	return (data[i]);
}

template <typename T> inline const T &TArray<T>::operator[](size_t i) const
{
	assert(i < size);
	return (data[i]);
}

template <typename T> inline void TArray<T>::push_back(const T &t)
{
	if (size >= capacity) {
		capacity = capacity ? 2 * capacity : 1;
		data = static_cast<T *>(
			safe_realloc(data, capacity * sizeof(T)));
	}
	data[size++] = t;
}

template <typename T> void TArray<T>::resize(size_t size)
{
	if (size > capacity) {
		data = static_cast<T *>(safe_realloc(data, size * sizeof(T)));
		capacity = size;
	}
	this->size = size;
}

template <typename T> void TArray<T>::reserve(size_t capacity)
{
	if (capacity > this->capacity) {
		data = static_cast<T *>(
			safe_realloc(data, capacity * sizeof(T)));
		this->capacity = capacity;
	}
}

template <typename T> inline void TArray<T>::clear()
{
	size = 0;
}
