#include <stdio.h>
#include "../common/cuda_helpers.c"

extern "C"{

#include "../common/cuda_data_structures.c"

}

#define THREADS  1024

typedef double(*deviceFunction)(double, double);

__device__ double max_of(double first, double next){

	return max(first,next);
}

 __device__ deviceFunction h_binary_op = max_of;


__device__ int set_boundary_conditions(Matrix a, Matrix c,const int x_index,const int y_index,const int global_index){

	// Boundary points descriptions
	// The top of the grid will have blockIdx.y == 0 and threadIdx.y == 0, so the sum must be 0
	// The left side of the grid have will blockIdx.x == 0 and threadIdx.x == so the sum must be 0
	// The right side must be MAX(blockIdx.x) e.g. (gridDim.x-1)  and have MAX(threadIdx.x) e.g. (blockDim.x -1) 


	if(y_index == 0 || y_index == a.num_rows  || x_index == 0 || x_index == a.num_columns ){
		c.elements[global_index] = a.elements[global_index];
		return 1;
	}
	return 0;
}

__global__ void laplace2d(Matrix a, Matrix c,deviceFunction binary_op){

	const int x_index = threadIdx.x + blockIdx.x*blockDim.x;
	const int y_index = threadIdx.y + blockIdx.y*blockDim.y; 
	const int global_index = x_index + a.num_columns*y_index;

	/*
	aaaa aaaa
	abbb bbba	
	abbb bbba
	abbb bbba
	
	abbb bbba
	abbb bbba
	abbb bbba
	aaaa aaaa

	*/
	if(!set_boundary_conditions(a,c,x_index,y_index,global_index))
	{
		if(x_index < a.num_columns && y_index < a.num_rows){
			c.elements[global_index] = a.elements[global_index -1] + a.elements[global_index +1] 
						 + a.elements[global_index + a.num_columns] + a.elements[global_index - a.num_columns];
		}

	}




}

int main(int argc, char *argv[]){

	printf("START\n");
	

	unsigned int num_elements = 1<<12;
	size_t size = sizeof(double) * num_elements;
	const int sqrt_threads = sqrt(THREADS);
	
	double *h_input = (double *)malloc(size);
	double *h_output   = (double *)malloc(size);
	
	int seed = time(NULL);
    	srand(seed);

	for(int i = 0;i < num_elements;i++){
		h_input[i] = (double)rand();
		h_output[i] = 0.0;
	}

	int num_rows = 64;
	int num_columns = 64;

	if(num_rows * num_columns != num_elements){

		printf("DIMENSION MISMATCH");
		return -1;
	}

	Matrix a = create_matrix(num_rows, num_columns,h_input);
	Matrix c = create_matrix(num_rows,num_columns,h_output);

	deviceFunction d_binary_op;

	CHECK_CUDA(cudaMemcpyFromSymbol(&d_binary_op, h_binary_op, sizeof(deviceFunction)));
	CHECK_CUDA(cudaDeviceSynchronize());


	dim3 blocks_dim(num_elements/sqrt_threads,num_elements/sqrt_threads);
	dim3 threads_dim(sqrt_threads,sqrt_threads);
	printf("START KERNEL\n");
	laplace2d<<<blocks_dim,threads_dim>>>(a, c,d_binary_op);
	cudaMemcpy(h_output,c.elements,size,cudaMemcpyDeviceToHost);

	printf("DESTROYING");
	destroy_matrix(a);
	destroy_matrix(c);
	free(h_input);
	free(h_output);
}


