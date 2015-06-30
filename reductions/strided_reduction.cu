
#include  "../common/cuda_helpers.c"

extern "C"{

#include "reduction_helpers.c"

}

__global__ void
vectorAdd(double *input,  double *output,int numElements)
{


	const int tid = threadIdx.x;
	double *offset = input + blockIdx.x * blockDim.x;

	if(tid +  blockIdx.x * blockDim.x >= numElements) return;


	for(int stride = blockDim.x/2; stride > 0; stride/=2){

		if(tid < stride){
		offset[tid] += offset[tid+stride];
		}

		__syncthreads();
	}
	if(tid==0){output[blockIdx.x] = offset[0];}
}



int
main(void)
{
	unsigned const int num_elements = 1<<13;
	const size_t size = num_elements*sizeof(double);

	double *orig_input;
	double *h_input;
	double *h_output;

	double *d_input;
        double *d_output;


	initialize_host(size,&h_input,&h_output);

	  for(int i = 0; i < num_elements;i++){
                h_input[i] = (double)i;
                h_output[i] = 0;
		printf("%f\n",h_input[i]);
        }

	initialize_device(size,&d_input,&d_output);
	copy_host_to_device(size,h_input,h_output,d_input,d_output);
	
	orig_input = (double *)malloc(size);
	memcpy(orig_input,h_input,size);

	int threadsPerBlock = 512;
	int blocksPerGrid =(num_elements + threadsPerBlock - 1) / threadsPerBlock;
    	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_input,d_output, num_elements);
        copy_device_to_host(size,h_input,h_output,d_input,d_output);
        check_reduction(orig_input,h_output,num_elements,blocksPerGrid);
	printf("DESTROYING\n");	
	destroy_host(h_input,h_output);
	free(orig_input);
}
