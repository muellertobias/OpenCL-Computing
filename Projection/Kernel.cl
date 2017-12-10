#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void Project(__global int *indexes)
{
	int i = get_global_id(0);

	int groupSize = get_local_size(0);
	int localID = get_local_id(0); 
	int groupID = get_group_id(0);

	int odd = groupID & 1;
	int offset = groupSize * (groupID - odd);

	indexes[i] = offset + localID * 2 + odd;
}  