#include <stdio.h>
#include "../common/cuda_helpers.c"

extern "C"{

#include "../common/cuda_data_structures.c"

}

#define THREADS  256

typedef double(*deviceFunction)(double, double);

__device__ double max_of(double first, double next){

	return max(first,next);
}

 __device__ deviceFunction h_binary_op = max_of;


__device__ bool valid_index(const int index,const int column_shift,const int row_shift, Matrix a,int *out){

	int base_index = index + column_shift*a.num_columns;

	//Right Edge
	if(base_index % a.num_columns == 0 && row_shift == -1 ){
		return false;

	}

	//Left Edge
	if((base_index + 1) % a.num_columns == 0 && row_shift == 1){
		return false;
	}

	int shifted_index = base_index + row_shift;
	*out = shifted_index;

	return (0 <= shifted_index && shifted_index < a.num_rows*a.num_columns);
}

__global__ void neigborhood_kernel(Matrix a, Matrix c,deviceFunction binary_op){

	int tid = threadIdx.x;
	int index = tid + blockDim.x*blockIdx.x;
	int local_max = a.elements[index];
	int current_index;


	for(int y_loop = -1;y_loop < 2;y_loop++){
		for(int x_loop = -1; x_loop < 2; x_loop++){
			
			if(valid_index(index,y_loop,x_loop,a,&current_index)){
				local_max = binary_op(local_max,a.elements[current_index]);
			}


		}
	}
	

	c.elements[index] = local_max;
}

int main(int argc, char *argv[]){

	printf("START\n");
	

	unsigned int num_elements = 1<<12;
	size_t size = sizeof(double) * num_elements;
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



	unsigned int num_blocks = (num_elements + THREADS - 1) / THREADS;
	printf("START KERNEL\n");
	neigborhood_kernel<<<num_blocks,THREADS>>>(a, c,d_binary_op);
	cudaMemcpy(h_output,c.elements,size,cudaMemcpyDeviceToHost);

	printf("DESTROYING");
	destroy_matrix(a);
	destroy_matrix(c);
	free(h_input);
	free(h_output);
}


