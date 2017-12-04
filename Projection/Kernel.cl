#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void OpenCLID(__global int *X)
{
    //Get our global thread ID
	int i = get_global_id(0);

	int groupSize = get_local_size(0);
	int ngroups = get_num_groups(0);
	int localID = get_local_id(0); 
	int groupID = get_group_id(0);

	//float base = 2.0f;
	int exponent = groupID - (groupID & 1);
	//int offset = (int)pow(base, exponent);
	int offset = exponent * groupSize;

	X[i] = offset + 2 * localID + (groupID & 1);
}  