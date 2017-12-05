#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void OpenCLID(__global int *X, const unsigned int sizeX, const unsigned int sizeY)
{
	int globalID = get_global_id(0);

	if (globalID < sizeX * sizeY) 
	{
		int localID = get_local_id(0); 
		if (localID % sizeX < sizeX/2){
			int column = ((localID % sizeX) + (localID % sizeX/sizeX))*(sizeY*sizeX/(sizeX/2));
			X[localID] = column + (localID / sizeX) * 2;
		} else {
			int id =  localID-sizeX/2;
			int column = ((id % sizeX) + (id % sizeX/sizeX))*(sizeY*sizeX/(sizeX/2));
			X[localID] = (column +(localID / sizeX) * 2)+1;
		}
		

		
	}
}  