#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void OpenCLMatrix(__global int *X)
{
    //Get our global thread ID
    int id = get_global_id(1);

	X[id] = id;
}  