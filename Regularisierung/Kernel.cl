#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void vecades(__global float *a, __global float *b,  __global float *c, const unsigned int n)
{
	int idx = get_global_id(0);

	if (idx < n) 
	{
		c[idx] = a[idx] + b[idx] * ((-1) + 2 * (idx & 1));
	} 
}  