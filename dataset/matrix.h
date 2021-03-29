#pragma once

template<typename T>
void matrix(T* arr,int row,int col);

template<typename T>
void zeros(T* arr, int row,int col);

template<typename T>
void eye(T* arr,int size);   // in identity matrix row=col

template<typename T>
void diagonal(T* arr,int size);

template<typename T>
void upper(T* arr,int size);

template<typename T>
void lower(T* arr, int size);

template<typename T>
void tridiagonal(T* arr,int size);

void test(int t);

