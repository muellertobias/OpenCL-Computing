// BandWidth.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <CL/opencl.h>
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string>
#include <sstream>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#define type int // float, double

using namespace std;

void testOpenCL(const char* kernelSource);
char* readSourceFile(const char* filename);

//using namespace std;

int main(int argc, char* argv[])
{
	const char* kernelSource = readSourceFile("KernelA.cl");

	// Initialize vectors on host
	printf_s("init...\n");
	testOpenCL(kernelSource);
	//int id = 0;

	printf("Press any key and then press enter...");
	return getchar();
}

void testOpenCL(const char* kernelSource)
{
	cl_mem d_values;				// Device output buffer
	cl_mem d_sums;

	cl_platform_id cpPlatform;		// OpenCL platform
	cl_device_id device_id;         // device ID
	cl_context context;				// context
	cl_command_queue queue;			// command queue
	cl_program program;				// program
	cl_kernel kernel;				// kernel

	size_t globalSize;
	cl_int err;

	// Host input array
	type *h_values;
	size_t n_values = 100000000UL;

	size_t bytes = n_values * sizeof(type);		// Size, in bytes, of each vector						  
	h_values = (type*)malloc(bytes);			// Allocate memory for each vector on host
	if (h_values == 0) 
	{
		printf_s("Out of memory\n");
		return;
	}

	memset(h_values, 1, bytes);

	// Host output array
	type *h_sums;
	size_t n_sums = 10000;
				  
	h_sums = (type*)malloc(n_sums * sizeof(type));			// Allocate memory for each vector on host

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);
	printf("GetPlatfrom: %d\n", err);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	printf("GetDeviceIDs: %d\n", err);

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
	kernel = clCreateKernel(program, "Sum", &err);
	printf("CreateKernel: %d\n", err);

	// Create the input array in device memory for our calculation
	d_values = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_values, CL_TRUE, 0, bytes, h_values, 0, NULL, NULL);
	printf("clEnqueueWriteBuffer: %d\n", err);

	// Set the arguments to our compute kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_values);
	err |= clSetKernelArg(kernel, 0, sizeof(unsigned int), &n_values); 
	err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_sums);
	err |= clSetKernelArg(kernel, 0, sizeof(unsigned int), &n_sums);
	printf("clSetKernelArg: %d\n", err);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &n_values, &n_sums, 0, NULL, NULL);
	printf("EnqueueNDRangeKernel: %d\n", err);
	if (err == CL_SUCCESS)
	{
		// Wait for the command queue to get serviced before reading back results
		clFinish(queue);

		// Read the results from the device
		clEnqueueReadBuffer(queue, d_sums, CL_TRUE, 0, bytes, h_sums, 0, NULL, NULL);

		// print some ids
		for (int i = 0; i < n_sums; i++)
		{
			cout << "Stelle: " << i << " id: " << h_sums[i] << endl;
		}
	}

	// release OpenCL resources
	clReleaseMemObject(d_values);
	clReleaseMemObject(d_sums);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	free(h_values);
	free(h_sums);
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
