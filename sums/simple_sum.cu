
#include  "../common/cuda_helpers.c"

extern "C"{

#include "sum_helpers.c"

}

__global__ void sum_kernel(double *a, double *b , double *output,int numElements)
{
	
	const int index = threadIdx.x  + blockIdx.x * blockDim.x;

	if(index >= numElements){return;}

	output[index] = a[index] + b[index];

}



int
main(void)
{
	unsigned const int num_elements = 1<<13;
	const size_t size = num_elements*sizeof(double);

	double *h_a_orig_input;
	double *h_b_orig_input;
	double *h_output;
	double *h_b_input;
	double *h_a_input;

        double *d_output;
	double *d_a_input;
	double *d_b_input;


	initialize_host(size,&h_a_input,&h_b_input,&h_output);

	  for(int i = 0; i < num_elements;i++){
                h_a_input[i] = (double)i;
		h_b_input[i] = (double)i;
                h_output[i] = 0;
        }

	initialize_device(size,&d_a_input,&d_b_input,&d_output);
	copy_host_to_device(size,h_a_input,h_b_input,h_output,d_a_input,d_b_input,d_output);
	
	h_a_orig_input = (double *)malloc(size);
	h_b_orig_input = (double *)malloc(size);

	memcpy(h_a_orig_input,h_a_input,size);
	memcpy(h_b_orig_input,h_b_input,size);
	
	int threadsPerBlock = DIM;
	int blocksPerGrid =(num_elements + threadsPerBlock - 1) / threadsPerBlock;
    	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    	sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a_input,d_b_input,d_output, num_elements);
        copy_device_to_host(size,h_a_input,h_b_input,h_output,d_a_input,d_b_input,d_output);
        check_sum(h_a_orig_input,h_b_orig_input,h_output,num_elements,blocksPerGrid);
	printf("DESTROYING\n");	
	destroy_host(h_a_input,h_b_input,h_output);
	destroy_device(d_a_input,d_b_input,d_output);
	free(h_a_orig_input);
	free(h_b_orig_input);
}
