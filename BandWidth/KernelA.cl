
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

		for (unsigned int i = n * id; i <= n * (id + 1) - 1; i++) 
		{
			sum += values[i];
		}

		sums[id] = sum; 
	}
}

