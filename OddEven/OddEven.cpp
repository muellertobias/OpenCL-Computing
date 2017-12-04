

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

//#include "stdafx.h"
#include <iostream>
#include <CL/opencl.h>
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string>
#include <sstream>

#define type int // float, double

using namespace std;

void testOpenCL(const char* kernelSource);
char* readSourceFile(const char* filename);
void printMatrix(const type * matrix, const size_t& n, const size_t& m);


int main(int argc, char* argv[])
{
	const char* kernelSource = readSourceFile("Kernel.cl");

	int id = 0;
	// Initialize vectors on host
	printf_s("init...\n");
	testOpenCL(kernelSource);
	//int id = 0;

	printf("Press any key and then press enter...");
	return getchar();
}

void testOpenCL(const char* kernelSource)
{

	// Device output buffer
	cl_mem d_X;

	cl_platform_id cpPlatform;		  // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;				  // context
	cl_command_queue queue;			  // command queue
	cl_program program;				  // program
	cl_kernel kernel;				  // kernel

	cl_int err;
	// Host output array
	type *h_matrix;

	// Number of work items in each local work group
	// use CL_DEVICE_MAX_WORK_ITEM_SIZES
	cl_uint max_work_item_dimensions;


	//printf("CL_DEVICE_MAX_WORK_ITEM_SIZES: ");

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);
	printf("GetPlatfrom: %d\n", err);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	printf("GetDeviceIDs: %d\n", err);

	// Das ware meien Idee
	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dimensions), &max_work_item_dimensions, NULL);
	size_t* max_work_item_sizes = (size_t*)malloc(sizeof(size_t) * max_work_item_dimensions);
	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_work_item_dimensions, max_work_item_sizes, NULL);
	printf("\n \n");

	size_t matrixSizeX = 4;
	size_t matrixSizeY = 3;

	size_t localSize = matrixSizeX * matrixSizeY;
	size_t globalSize = matrixSizeX * matrixSizeY;

	size_t bytes = globalSize * sizeof(type);  // Size, in bytes
					 
	h_matrix = (type*)malloc(bytes); // Allocate memory
	memset(h_matrix, NULL, bytes);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	printf("CreateContext: %d\n", err);

	// Create a command queue 
	//queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);
	queue = clCreateCommandQueue(context, device_id, NULL, &err);
	printf("CreateCommandQueue: %d\n", err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)& kernelSource, NULL, &err);
	printf("CreateProgramWithSource: %d\n", err);

	// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "OpenCLID", &err);
	printf("CreateKernel: %d\n", err);

	// Create the input and output arrays in device memory for our calculation
	d_X = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_X, CL_TRUE, 0, bytes, h_matrix, 0, NULL, NULL);

	// Set the arguments to our compute kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_X);
	err = clSetKernelArg(kernel, 1, sizeof(unsigned int), &matrixSizeX);
	err = clSetKernelArg(kernel, 2, sizeof(unsigned int), &matrixSizeY);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, d_X, CL_TRUE, 0, bytes, h_matrix, 0, NULL, NULL);
	if (err == CL_SUCCESS) 
		printMatrix(h_matrix, matrixSizeX, matrixSizeY);

	// release OpenCL resources
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	free(h_matrix);
}

char* readSourceFile(const char* filename)
{
	FILE *fp;
	fopen_s(&fp, filename, "rb");
	fseek(fp, 0, SEEK_END);
	long size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	char* source = (char*)malloc(size + 1);
	fread(source, size, 1, fp);
	fclose(fp);
	source[size] = 0;
	return source;
}

void printMatrix(const type * matrix, const size_t& n, const size_t& m)
{
	cout << endl;
	for (size_t i = 0; i < n * m; i++)
	{
		cout << matrix[i] << " ";
		if (i % n == n - 1 && i > 0)
			cout << endl;
	}

	cout << endl;
}



















