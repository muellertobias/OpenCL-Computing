// POSIX_MVM.cpp : Defines the entry point for the console application.
// Julian Kirsch & Tobias Müller

#include <cstdlib>
#include <iostream>
#include <ctime>
#include <omp.h>
#include <pthread.h> 

using namespace std;

#define type int

typedef struct
{
	int threadID;
	type** matrixA;
	type* vectorB;
	type* vectorC;
	size_t nValues;
	int nThreads;
	double runtime_s;
} ThreadParameters;

void* threaded_add(void* parameters)
{
	ThreadParameters* param = (ThreadParameters*)parameters;

	type** A = param->matrixA;
	type* c = param->vectorC;
	type* b = param->vectorB;

	clock_t startTime = clock();

	const int nRows = (int)((double)param->nValues / (double)param->nThreads);

	int* rows = (int*)malloc(nRows * sizeof(int));
	memset(rows, 0, nRows * sizeof(int));

	for (size_t i = 0; i < nRows / 2; i++)
	{
		rows[i] = param->threadID * nRows + i;
		rows[nRows - i - 1] = param->nValues - 1 - rows[i];
	}

	for (size_t row = 0; row < nRows; row++)
	{
		int i = rows[row];
		type c_i = 0;
		for (size_t j = i; j < param->nValues; j++)
		{
			c_i += A[i][j] * b[j];
		}
		c[i] = c_i;
	}

	param->runtime_s += (double)(clock() - startTime) / CLOCKS_PER_SEC;
	free(rows);
	return 0;
}


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

void initVectorWithValue(type* vector, int n, type value)
{
#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		vector[i] = value;
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
	const size_t n = 16384;//8192; // nur 2er Potenzen erlaubt!
	const size_t threadCount = 4; 

	type **a = new type*[n];
	type b[n];
	type c[n];

	cout << "Init..." << endl;
	initVectorWithValue(b, n, 1);
	initVectorWithValue(c, n, 0);
	initMatrix(a, n, n);


	cout << "Calculate A x b = c ..." << endl;
	//-----Thread-----
	pthread_t thr[threadCount];
	ThreadParameters params[threadCount];

	clock_t startThread = clock();
	for (size_t i = 0; i < threadCount; i++)
	{
		params[i].threadID = i;
		params[i].nThreads = threadCount;
		params[i].nValues = n;
		params[i].matrixA = a;
		params[i].vectorB = b;
		params[i].vectorC = c;
		params[i].runtime_s = 0;

		pthread_create(&thr[i], NULL, threaded_add, (void*)&params[i]);
	}

	for (size_t i = 0; i < threadCount; i++)
	{
		pthread_join(thr[i], NULL); // wartet auf Threads
	}

	double endThread = (double)(clock() - startThread) / CLOCKS_PER_SEC;
	cout << "Time with threads: " << endThread << endl;

	for (size_t i = 0; i < threadCount; i++)
	{
		cout << i << ": " << params[i].runtime_s << " s" << endl;
	}

	// Sequenziell
	ThreadParameters param;

	param.threadID = 0;
	param.nThreads = 1;
	param.nValues = n;
	param.matrixA = a;
	param.vectorB = b;
	param.vectorC = c;
	param.runtime_s = 0;

	startThread = clock();
	threaded_add((void*)&param);
	endThread = (double)(clock() - startThread) / CLOCKS_PER_SEC;
	cout << "Time single thread: " << endThread << endl;
	cout << 0 << ": " << param.runtime_s << " s" << endl;
	cout << "Press any key..." << endl;

	getchar();

	return 0;
}

