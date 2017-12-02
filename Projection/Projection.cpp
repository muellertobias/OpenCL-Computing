

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

void testOpenCL(const char* kernelSource, type* h_X, size_t n, const size_t bytes, size_t localSize);
void callGPU(cl_event& event, cl_int &err, const cl_command_queue &queue, const size_t &bytes, const cl_kernel &kernel, cl_mem &d_X, size_t &globalSize, size_t &localSize, type *h_X, const size_t &n);
char* readSourceFile(const char* filename);

//using namespace std;

int main(int argc, char* argv[])
{
	const char* kernelSource = readSourceFile("Kernel.cl");

	// Host output array
	type *h_X;
	

	size_t n = 1;

	// Number of work items in each local work group
	size_t localSize = 64;

	size_t bytes = n * sizeof(type);  // Size, in bytes, of each vector

	// Allocate memory for each vector on host
	h_X = (type*) malloc(bytes);
	memset(h_X, 100, bytes);
	int id = 0;
	// Initialize vectors on host
	printf_s("init...\n");
	testOpenCL(kernelSource, h_X, n, bytes, localSize);
	//int id = 0;

	/*for (int i = 0; i < bytes;i++) {
		id = h_X[i];
		cout << "Stelle: " <<i << " id: " << id << endl;
	}*/
	//release host memory
	free(h_X);

	printf("Press any key and then press enter...");
	return getchar();
}

void testOpenCL(const char* kernelSource, type* h_X, size_t n, const size_t bytes, size_t localSize)
{

	// Device output buffer
	cl_mem d_X;

	cl_platform_id cpPlatform;		  // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;				  // context
	cl_command_queue queue;			  // command queue
	cl_program program;				  // program
	cl_kernel kernel;				  // kernel

	size_t globalSize;
	cl_int err;

	// use CL_DEVICE_MAX_WORK_ITEM_SIZES
	cl_uint max_work_item_dimensions;
	

	//printf("CL_DEVICE_MAX_WORK_ITEM_SIZES: ");

	// Number of total work items - localSize must be devisor
	globalSize = ceil(n / (float)localSize) * localSize;

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
	printf("CL_DEVICE_MAX_WORK_ITEM_SIZES: ");
	n = 1;
	for (size_t i = 0; i < max_work_item_dimensions; ++i) {
		printf("%lu \\ ", max_work_item_sizes[i]);
		n *= max_work_item_sizes[i];
	}
	printf("\n");
	printf("Thread in one Dimesion: %lu \n \n \n", n);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	printf("CreateContext: %d\n", err);


	cl_event event;

	// Create a command queue 
	queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);
	//queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
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

	callGPU(event, err, queue, bytes, kernel, d_X, globalSize, localSize, h_X, n);

	cl_ulong time_start, time_end;
	double total_time;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("OpenCL Execution time: %f ms\n", total_time / 1000000.0);

	// release OpenCL resources
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
} 


void callGPU(cl_event& event, cl_int &err, const cl_command_queue &queue, const size_t &bytes, const cl_kernel &kernel, cl_mem &d_X, size_t &globalSize, size_t &localSize, type *h_X, const size_t &n)
{
	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, d_X, CL_TRUE, 0, bytes, h_X, 0, NULL, NULL);

	// Set the arguments to our compute kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_X);
	err |= clSetKernelArg(kernel, 1, sizeof(unsigned int), &n);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, &event);
	clWaitForEvents(1, &event);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, d_X, CL_TRUE, 0, bytes, h_X, 0, NULL, NULL);
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



















