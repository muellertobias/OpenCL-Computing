
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void Sum(__global int *values, const unsigned int n_values, __global int *sums, const unsigned int n_sums)
{
    //Get our global thread ID
    int id = get_global_id(0);
	
    //Make sure we do not go out of bounds
	if (id < n)
	{
		int n = n_values / n_sums;
		int sum = 0;
		int offset = id * n;

		for (unsigned int i = 0; i < n; i++) 
		{
			sum += values[offset + i];
		}

		sums[id] = sum; 
	}
}
