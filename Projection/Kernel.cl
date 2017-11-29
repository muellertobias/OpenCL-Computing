#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void OpenCLMatrix(__global int *A, __global int *b, __global int *c, const unsigned int n)
{
    //Get our global thread ID
    int id = get_global_id(0);

    //Make sure we do not go out of bounds
	if (id < n)
	{
		c[id] = 0;
		for (int i = 0; i < n; i++) 
		{
			c[id] += A[id * n + i] * b[i];
		}
	}
}  