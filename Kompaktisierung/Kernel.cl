#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void mandelbrot(__global int* matrix, const unsigned int width, const unsigned int height)
{
	int index = get_global_id(0);

	if (index < width * height) 
	{
		int counter = 0;
		int x = index % width;				// Spalte
		int y = (index - x) / height;		// Zeile

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

// Programme vom Keller um Mandelbrot Bild zu erzeugen
/*calcmandel(int *a,int N,int M){

	int x,y,i,cnt; 
	complex z;
	x=proj(blockIdx.x,threadIdx.x);
	y=proj(blockIdx.y,threadIdx.y);
	z=startwert(x,y); 
	cnt=0;

	for(i=0;i<255;i++)
	{
		if(cbetrag(z)<2)
		{ 
			z=F(z); 
			cnt++; 
		}
	}
	a[x][y] = cnt; 
}

calcmandel2(int *a,int N,int M)
{
	int idx,x,y,i,cnt; 
	complex z;
	idx=proj(blockIdx.x,threadIdx.x);
	x=listex[idx]; 
	y=listey[idx];
	z=startwert(x,y);
	for(i=0;i<16;i++) z=F(z);
	cnt=16;

	for(i=16;i<255;i++)
	{
		if(cbetrag(z)<2)
		{ 
			z=F(z); 
			cnt++; 
		}
	}
	a[x][y] = cnt; 
}*/