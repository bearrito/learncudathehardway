
#include  "../common/cuda_helpers.c"

extern "C"{

#include "reduction_helpers.c"

}

#define DIM 512

__global__ void
reduce(double *input,  double *output,const unsigned int numElements)
{
	const int tid = threadIdx.x;
	__shared__ double cheap[DIM]; 

	if(tid + blockIdx.x * blockDim.x >= numElements) return; 
	
	double *offset = input + blockIdx.x * blockDim.x;
	
	cheap[tid] = offset[tid];
	__syncthreads();

//	printf("BLOCK[%d] TID[%d]\n",blockIdx.x,tid);
	if(tid < 512 && blockDim.x >= 1024){cheap[tid] += cheap[tid + 512];}
        __syncthreads();

	if(tid < 256 && blockDim.x >= 512){cheap[tid] += cheap[tid + 256];}
        __syncthreads();

	if(tid < 128 && blockDim.x >= 256){cheap[tid] += cheap[tid + 128];}
        __syncthreads();

	if(tid < 64 && blockDim.x >= 128){cheap[tid] += cheap[tid + 64];}
	__syncthreads();

	if(tid < 32){

	volatile double *vol_temp = cheap;
	vol_temp[tid] += vol_temp[tid + 32];
	vol_temp[tid] += vol_temp[tid + 16];
	vol_temp[tid] += vol_temp[tid + 8];
	vol_temp[tid] += vol_temp[tid + 4];
        vol_temp[tid] += vol_temp[tid + 2];
        vol_temp[tid] += vol_temp[tid + 1];
	}

	if(tid == 0 ){output[blockIdx.x] = cheap[0];}

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
        }

	initialize_device(size,&d_input,&d_output);
	copy_host_to_device(size,h_input,h_output,d_input,d_output);
	
	orig_input = (double *)malloc(size);
	memcpy(orig_input,h_input,size);

	int threadsPerBlock = DIM;
	int blocksPerGrid =(num_elements + threadsPerBlock - 1) / threadsPerBlock;
    	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    	reduce<<<blocksPerGrid, threadsPerBlock>>>(d_input,d_output, num_elements);
        copy_device_to_host(size,h_input,h_output,d_input,d_output);
        check_reduction(orig_input,h_output,num_elements,blocksPerGrid);
	printf("DESTROYING\n");	
	destroy_host(h_input,h_output);
	free(orig_input);
}
