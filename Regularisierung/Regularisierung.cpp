

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

	// Device output buffer
	cl_mem d_a;
	cl_mem d_b;
	cl_mem d_c;

	cl_platform_id cpPlatform;		  // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;				  // context
	cl_command_queue queue;			  // command queue
	cl_program program;				  // program
	cl_kernel kernel;				  // kernel

	cl_int err;
	// Host output array
	type *h_a;
	type *h_b;
	type *h_c;

	size_t n = 1;

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

	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dimensions), &max_work_item_dimensions, NULL);
	size_t* max_work_item_sizes = (size_t*)malloc(sizeof(size_t) * max_work_item_dimensions);
	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_work_item_dimensions, max_work_item_sizes, NULL);
	
	size_t vectorSize = 16;

	size_t localSize = vectorSize;
	size_t globalSize = vectorSize;

	size_t bytes = vectorSize * sizeof(type);  // Size, in bytes, of each vector

									  // Allocate memory for each vector on host
	h_a = (type*)malloc(bytes);
	h_b = (type*)malloc(bytes);

	for (int i = 0;i < vectorSize;i++) {
		h_a[i] = 1;
		h_b[i] = 1;
	}
	h_c = (type*)malloc(bytes);
	memset(h_c, NULL, bytes);

	// Create a context  
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	printf("CreateContext: %d\n", err);


	cl_event event;

	// Create a command queue 
	queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);
	//queue = clCreateCommandQueue(context, device_id, NULL, &err);
	printf("CreateCommandQueue: %d\n", err);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)& kernelSource, NULL, &err);
	printf("CreateProgramWithSource: %d\n", err);

	// Build the program executable 
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "vecades", &err);
	printf("CreateKernel: %d\n", err);

	// Create the input and output arrays in device memory for our calculation
	d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, bytes, h_a, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, bytes, h_b, 0, NULL, NULL);


	size_t l = max_work_item_sizes[0];
	//size_t g = 2048;
	// Set the arguments to our compute kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
	err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &vectorSize);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &n, &l, 0, NULL, &event);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);

	for (int i = 0; i < vectorSize;i++) {
		printf("%d\n", h_c[i]);
	}
	// release OpenCL resources
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	free(h_a);
	free(h_b);
	free(h_c);
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



















