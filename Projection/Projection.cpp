
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

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

#define type int // float, double

using namespace std;

void testOpenCL(const char* kernelSource);
char* readSourceFile(const char* filename);

//using namespace std;

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
	
	cl_mem d_indexes;						// Device output buffer

	cl_platform_id cpPlatform;		// OpenCL platform
	cl_device_id device_id;         // device ID
	cl_context context;				// context
	cl_command_queue queue;			// command queue
	cl_program program;				// program
	cl_kernel kernel;				// kernel

	size_t globalSize;
	cl_int err;

	// Host output array
	type *h_indexes;
	size_t nIndexes = 1;

	
	// use CL_DEVICE_MAX_WORK_ITEM_SIZES
	cl_uint max_work_item_dimensions;
	
	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);
	printf("GetPlatfrom: %d\n", err);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	printf("GetDeviceIDs: %d\n", err);

	// Device Info
	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dimensions), &max_work_item_dimensions, NULL);

	size_t* max_work_item_sizes = (size_t*)malloc(sizeof(size_t) * max_work_item_dimensions);
	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_work_item_dimensions, max_work_item_sizes, NULL);
	
	printf("\n\nCL_DEVICE_MAX_WORK_ITEM_SIZES: ");
	for (size_t i = 0; i < max_work_item_dimensions; ++i)
	{
		printf("%lu \\ ", max_work_item_sizes[i]);
		nIndexes *= max_work_item_sizes[i];
	}

	// Number of work items in each local work group
	size_t localSize = max_work_item_sizes[0];

	printf("\n");
	printf("Work item in one Dimesion: %lu \n", nIndexes);
	printf("Work item in one work group: %lu \n \n \n", localSize);

	size_t bytes = nIndexes * sizeof(type);		// Size, in bytes, of each vector						  
	h_indexes = (type*)malloc(bytes);			// Allocate memory for each vector on host
	memset(h_indexes, NULL, bytes);

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
	kernel = clCreateKernel(program, "Project", &err);
	printf("CreateKernel: %d\n", err);

	// Create the input array in device memory for our calculation
	d_indexes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_indexes, CL_TRUE, 0, bytes, h_indexes, 0, NULL, NULL);

	// Set the arguments to our compute kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_indexes);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &nIndexes, &localSize, 0, NULL, NULL);
	printf("EnqueueNDRangeKernel: %d\n", err);
	if (err == CL_SUCCESS) 
	{
		// Wait for the command queue to get serviced before reading back results
		clFinish(queue);

		// Read the results from the device
		clEnqueueReadBuffer(queue, d_indexes, CL_TRUE, 0, bytes, h_indexes, 0, NULL, NULL);

		// print some ids
		for (int i = 0; i < 4 * localSize; i++)
		{
			if ((i % localSize) == 0)
			{
				printf("-----------------------------------------------\n");
			}
			cout << "Stelle: " << i << " id: " << h_indexes[i] << endl;
		}

		// print last two ids
		cout << "Stelle: " << nIndexes - 2 << " id: " << h_indexes[nIndexes - 2] << endl;
		cout << "Stelle: " << nIndexes - 1 << " id: " << h_indexes[nIndexes - 1] << endl;
	}

	// release OpenCL resources
	clReleaseMemObject(d_indexes);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	free(h_indexes);
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



















