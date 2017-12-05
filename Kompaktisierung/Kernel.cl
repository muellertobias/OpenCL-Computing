#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void mandelbrot(__global int* matrix, const unsigned int width, const unsigned int height)
{
	int index = get_global_id(0);

	if (index < width * height) 
	{
		int counter = 0;
		int x = index % width;
		int y = (index - x) / height;

		float cReal = x;
		float cImag = y;

		for (int i = 0; i < 256; ++i) 
		{
			if ((cReal * cReal + cImag * cImag) <= 4.0f) 
			{
				float re = cReal;
				float im = cImag;

				cReal = re * re - im * im;
				cImag = 2.0f * re * im;

				counter++;
			}
		}

		matrix[index] = counter;
	} 
}  