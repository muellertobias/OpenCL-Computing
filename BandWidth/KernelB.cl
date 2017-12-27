
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void sum(__global int *values, const unsigned int n_values, __global int *sums, const unsigned int n_sums)
{
    //Get our global thread ID
    int id = get_global_id(0);
	int n = n_values / n_sums;

    //Make sure we do not go out of bounds
	if (id < n)
	{
		int sum = 0;

		for (unsigned int j = 0; j < n; j++) 
		{
			sum += values[j + id * n];
		}

		sums[id] = sum; 
	}
}

