// OpenMP_MVM.cpp : Defines the entry point for the console application.
// Julian Kirsch & Tobias Müller

#include "stdafx.h"
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <omp.h>

using namespace std;

#define type int

void initVectorWithRandom(type* vector, int n)
{
	srand(time(NULL));
	#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		//vector[i] = (type)rand() / (type)(RAND_MAX / (type)rand());
		vector[i] = i;
	}
}

void initVectorWithNull(type* vector, int n)
{
	#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		vector[i] = 0;
	}
}

void initMatrixWithNull(type** matrix, int n, int m)
{
	#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		matrix[i] = new type[n];
		for (int j = 0; j < n; j++)
		{
			matrix[i][j] = 0;
		}
	}
}

/*
	| 1 2 3 |
	| 0 4 5 |
	| 0 0 6 |
*/
void initMatrix(type** matrix, int n, int m)
{
	#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		matrix[i] = new type[n];
		memset(matrix[i], 0, n * sizeof(type));

		for (int j = i; j < n; j++)
		{
			matrix[i][j] = i * n + j + 1;
		}
	}
}

int main()
{
	const int n = 20000;

	type **a = new type*[n];
	type b[n];
	type c[n];

	cout << "Init..." << endl;
	initVectorWithRandom(b, n);
	initVectorWithNull(c, n);
	initMatrix(a, n, n);

	cout << "Calculate A x b = c ..." << endl;
	clock_t start = clock();

	double times[4];
	for (int i = 0; i < 4; i++)
	{
		times[i] = 0.0;
	}

	#pragma omp parallel
	for (int i = 0; i < n; i++)
	{
		double wtime = omp_get_wtime();
		type c_i = 0;

		#pragma omp for schedule(guided)
		for (int j = i; j < n; j++)
		{
			c_i += a[i][j] * b[j];
		}

		c[i] = c_i;

		wtime = omp_get_wtime() - wtime;
		times[omp_get_thread_num()] += wtime;
	}

	for (int i = 0; i < 4; i++)
	{
		cout << times[i] << endl;
	}

	clock_t stop = clock();
	clock_t difference = stop - start;
	double t = (double)difference / CLOCKS_PER_SEC;
	cout << "Finished: " << t << "s" << endl; 
	cout << "Press any key..." << endl;
	
	getchar();

	return 0;
}

