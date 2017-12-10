#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void OpenCLID(__global int *matrix, const unsigned int sizeX, const unsigned int sizeY)
{
	int globalID = get_global_id(0);

	if (globalID < sizeX * sizeY) 
	{
		int localID = get_local_id(0); 

		int column = (localID % sizeX) * (sizeX + 2) / sizeX * 2 * sizeY;

		if (localID % sizeX >= sizeX / 2) 
		{
			column = column - (sizeX * sizeY) - (2 * sizeY) + 1;
		}

		matrix[localID] = column + (localID / sizeX) * 2;
	}
}  