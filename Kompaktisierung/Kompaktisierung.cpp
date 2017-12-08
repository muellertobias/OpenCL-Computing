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
#include <Windows.h>

#define type int // float, double

using namespace std;

void calcMandel(type* matrix, int width, int height);
void print(type* matrix, int width, int height);
void print(type* values, int n);
void colorPrint(type* matrix, int width, int height);
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
	size_t width = 125;
	size_t height = width;

	// Device output buffer
	cl_mem d_matrix;
	cl_mem d_listX;
	cl_mem d_listY;

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

	size_t localSize = height;
	size_t globalSize = globalSize = ceil(height / (float)localSize) * localSize;;
	int bytes = height * width * sizeof(type);
	
	// Allocate memory for each vector on host
	type* matrix = (type*)malloc(bytes);
	memset(matrix, 0, bytes);

	calcMandel(matrix, width, height);

	int* listX = (int*)malloc(bytes);
	memset(listX, 0, bytes);

	int* listY = (int*)malloc(bytes);
	memset(listY, 0, bytes);

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
	kernel = clCreateKernel(program, "compact", &err);
	printf("CreateKernel: %d\n", err);

	// Create the input and output arrays in device memory for our calculation
	d_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_listX = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
	d_listY = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_matrix, CL_TRUE, 0, bytes, matrix, 0, NULL, NULL);
	printf("clEnqueueWriteBuffer: %d\n", err);

	// Set the arguments to our compute kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_matrix);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_listX);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_listY);
	err |= clSetKernelArg(kernel, 3, bytes, NULL);

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	printf("EnqueueNDRangeKernel: %d\n", err);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	clEnqueueReadBuffer(queue, d_listX, CL_TRUE, 0, bytes, listX, 0, NULL, NULL);
	clEnqueueReadBuffer(queue, d_listY, CL_TRUE, 0, bytes, listY, 0, NULL, NULL);

	// print listX and listY
	printf("Lists:\n");
	for (int i = 0; i < height * width; i++)
	{
		if (listX[i] != 0 && listY[i] != 0 && i > 0)
			printf("%d %d\n", listX[i], listY[i]);
	}

	colorPrint(matrix, width, height);

	// release OpenCL resources
	clReleaseMemObject(d_matrix);
	clReleaseMemObject(d_listX);
	clReleaseMemObject(d_listY);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	free(matrix);
	free(listX);
	free(listY);
}

void calcMandel(type* matrix, int width, int height) 
{
	for (int index = 0; index < width * height; index++) 
	{
		int counter = 0;
		float x = ((index % width) - (width / 2.0f)) / (width / 2.0f);				// Spalte
		float y = ((index - x) / height - (height / 2.0f)) / (height / 2.0f);		// Zeile

		x -= 0.5f; // Verschiebung nach links

		float cReal = 0.0f;
		float cImag = 0.0f;

		for (int i = 0; i < 16; ++i)
		{
			if ((cReal * cReal + cImag * cImag) <= 4.0f)
			{
				float re = cReal;
				float im = cImag;

				cReal = re * re - im * im + x;
				cImag = 2.0f * re * im + y;

				counter++;
			}
		}

		matrix[index] = counter;
	}
}

void print(type* values, int n)
{
	for (int i = 0; i < n; i++)
	{
		printf("%d\n", values[i]);
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

void colorPrint(type* matrix, int width, int height) 
{
	HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			SetConsoleTextAttribute(console, matrix[i * width + j]);
			printf("0");
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

