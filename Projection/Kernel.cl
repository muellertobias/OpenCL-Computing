#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void OpenCLID(__global int *X)
{
	int i = get_global_id(0);

	int groupSize = get_local_size(0);
	int localID = get_local_id(0); 
	int groupID = get_group_id(0);

	int exponent = groupID - (groupID & 1);
	int offset = exponent * groupSize;

	X[i] = offset + 2 * localID + (groupID & 1);
}  