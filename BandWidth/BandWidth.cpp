// BandWidth.cpp : Defines the entry point for the console application.
//

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
void printError(const char* functionName, int error)
{
	if (error != 0)
		printf_s("%s: %d", functionName, error);
}

//using namespace std;

int main(int argc, char* argv[])
{
	const char* kernelSource = readSourceFile("KernelA.cl");

	// Initialize vectors on host
	printf_s("init...\n");
	for (int i = 0; i < 20; i++) 
	{
		testOpenCL(kernelSource);
	}

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

	cl_event event;

	size_t globalSize;
	cl_int err;

	// Host input array
	type *h_values;
	size_t n_values = 100000000UL;

	size_t bytes_values = n_values * sizeof(type);		// Size, in bytes, of each vector						  
	h_values = (type*)malloc(bytes_values);			// Allocate memory for each vector on host
	if (h_values == 0) 
	{
		printf_s("Out of memory\n");
		return;
	}

	memset(h_values, 0, bytes_values);
	for (size_t i = 0; i < n_values; i++)
	{
		h_values[i] = i % 10000;
	}

	size_t localSize = 256;
	globalSize = ceil(n_values / (float)localSize) * localSize;

	// Host output array
	type *h_sums;
	size_t n_sums = 10000;

	size_t bytes_sums = n_sums * sizeof(type);
	h_sums = (type*)malloc(n_sums * sizeof(type));			// Allocate memory for each vector on host

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);
	printError("GetPlatfrom: %d\n", err);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	printError("GetDeviceIDs: %d\n", err);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	printError("CreateContext %d\n", err);

	// Create a command queue 
	//queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);
	queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	printError("CreateCommandQueue: %d\n", err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)& kernelSource, NULL, &err);
	printError("CreateProgramWithSource: %d\n", err);

	// Build the program executable 
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	printError("BuildProgram: %d\n", err);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "sum", &err);
	printError("CreateKernel: %d\n", err);

	// Create the input array in device memory for our calculation
	d_values = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_values, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_values, CL_TRUE, 0, bytes_values, h_values, 0, NULL, NULL);
	printError("EnqueueWriteBuffer: %d\n", err);

	d_sums = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_sums, NULL, NULL);

	// Set the arguments to our compute kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_values);
	err |= clSetKernelArg(kernel, 1, sizeof(unsigned int), &n_values); 
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_sums);
	err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n_sums);
	printError("SetKernelArg: %d\n", err);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, &event);
	printError("EnqueueNDRangeKernel: %d\n", err);

	clWaitForEvents(1, &event);
	if (err == CL_SUCCESS)
	{
		// Wait for the command queue to get serviced before reading back results
		clFinish(queue);

		cl_ulong time_start, time_end;
		double total_time;
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		total_time = time_end - time_start;
		printf_s("OpenCL Execution time: %f ms\n", total_time / 1000000.0);

		// Read the results from the device
		clEnqueueReadBuffer(queue, d_sums, CL_TRUE, 0, bytes_sums, h_sums, 0, NULL, NULL);

		// print some ids
		/*for (int i = 0; i < n_sums; i++)
		{
			cout << "Stelle: " << i << " id: " << h_sums[i] << endl;
		}*/
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
