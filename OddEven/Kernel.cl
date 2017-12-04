#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void OpenCLID(__global int *X, const unsigned int sizeX, const unsigned int sizeY)
{
	int globalID = get_global_id(0);

	if (globalID < sizeX * sizeY) 
	{
		int localID = get_local_id(0); 
		//int numGroups = get_num_groups(0);
		//int groupSizeX = get_local_size(0);
		//int groupSizeY = get_local_size(1);

		int columnOffset = (globalID & 1) * sizeX / 2;

		//int indexX = columnOffset + globalID + (globalID % sizeY) * sizeX;
		//int indexY = 
		int index = globalID * (sizeX / 2);
		//X[indexX + indexY * sizeX] = globalID;
		X[index % sizeX * sizeY] = globalID;
	}
}  