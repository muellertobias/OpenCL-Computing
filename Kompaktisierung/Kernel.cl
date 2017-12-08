#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void compact(__global int* matrix, __global int* listX, __global int* listY, __local int* num)
{
	int rowIndex = get_global_id(0);
	const int size = get_local_size(0);

	int counter = 0;
	for (int column = 0; column < size; column++) 
	{
		if (matrix[rowIndex * size + column] == 16)  
		{
			counter++;
		}
	}

	num[rowIndex] = counter;

	barrier(CLK_LOCAL_MEM_FENCE);
	if (rowIndex == 0) 
	{
		int sum = 0;
		int sumold = 0;

		for (int i = 0; i < size; i++) 
		{ 
			sum += num[i]; 
			num[i] = sumold; 
			sumold = sum; 
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int index = num[rowIndex];

	for (int column = 0; column < size; column++) 
	{
		if (matrix[rowIndex * size + column] == 16)  
		{
			listX[index] = column;
			listY[index] = rowIndex;
			index++;
		}
	}
}  
