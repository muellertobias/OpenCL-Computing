#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void OpenCLID(__global int *X, const unsigned int n)
{
    //Get our global thread ID
	int id = get_global_id(0);
	X[id] = id;

	
}  