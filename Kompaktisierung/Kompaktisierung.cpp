#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

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

void print(type* matrix, int width, int height);
void print(type* values, int n);
void testOpenCL(const char* kernelSource);
char* readSourceFile(const char* filename);

int main(int argc, char* argv[])
{
	const char* kernelSource = readSourceFile("Kernel.cl");

	// Initialize vectors on host
	printf_s("init...\n");
	testOpenCL(kernelSource);

	printf("Press any key and then press enter...");
	return getchar();
}

void testOpenCL(const char* kernelSource)
{
	size_t width = 15;
	size_t height = 10;

	// Device output buffer
	cl_mem d_input;
	cl_mem d_output;

	cl_platform_id cpPlatform;		  // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;				  // context
	cl_command_queue queue;			  // command queue
	cl_program program;				  // program
	cl_kernel kernel;				  // kernel

	cl_int err;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);
	printf("GetPlatfrom: %d\n", err);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	printf("GetDeviceIDs: %d\n", err);


	size_t size = height * width;
	size_t localSize = 256;
	size_t globalSize = 256;
	int bytes = height * width * sizeof(int);

	// Allocate memory for each vector on host
	int* matrix = (int*)malloc(bytes);
	memset(matrix, 0, bytes);

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
	kernel = clCreateKernel(program, "mandelbrot", &err);
	printf("CreateKernel: %d\n", err);

	// Create the input and output arrays in device memory for our calculation
	d_input = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, bytes, matrix, 0, NULL, NULL);
	printf("clEnqueueWriteBuffer: %d\n", err);

	// Set the arguments to our compute kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
	err |= clSetKernelArg(kernel, 1, sizeof(unsigned int), &width);
	err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &height);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	printf("EnqueueNDRangeKernel: %d\n", err);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, d_input, CL_TRUE, 0, bytes, matrix, 0, NULL, NULL);

	print(matrix, width, height);

	// release OpenCL resources
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	free(matrix);
}

void print(type* values, int n)
{
	for (int i = 0; i < n; i++)
	{
		printf("%f\n", values[i]);
	}
}

void print(type* matrix, int width, int height)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			printf("%d ", matrix[i * width + j]);
		}
		printf("\n");
	}
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

