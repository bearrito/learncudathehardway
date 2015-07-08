#include <stdio.h>
#include "../common/cuda_helpers.c"


#define THREADS  64

__global__ void belloch_scan_kernel(double *scannable,double *scanned,const int total_num_elements){


	const int num_elements = 2 * THREADS;
	__shared__ double local[num_elements];
	const int tid = threadIdx.x;
	const int block_shift = 2*blockDim.x *blockIdx.x;
	double *shifted_data = scannable + block_shift;

	if(2*blockDim.x != num_elements){

		printf("KERNEL BLOCKSIZE MISCONFIG");

	}

	if(blockDim.x != THREADS){

		printf("BLOCK SIZE NOT EQUAL THREAD COUNT");

	}


	local[2*tid] = shifted_data[2*tid];
	local[2*tid+1] = shifted_data[2*tid+1];
	__syncthreads();
	int stride=1;


    	for (int d = blockDim.x; d > 0; d >>= 1)
    	{   
    		__syncthreads();  
       		if (tid < d)  
       		{ 

			 
			int virtual_zero_index = (stride*2*tid) + stride -1;
    			int active_site = virtual_zero_index+stride;  
			local[active_site] += local[virtual_zero_index];  
    		}  
    		stride *= 2;  	
	}

         
    	if (tid == 0) { local[2*blockDim.x - 1] = 0.0; }
	

	
	
	for(int d = 1; d < 2*blockDim.x; d<<=1){

		stride >>= 1;
		__syncthreads();
		
		if(tid < d)
		{

			int swap_index = (stride*2*tid) + stride -1;
			int add_index = swap_index+stride;

			if(swap_index == 0 && blockIdx.x == 0){
				printf("si = %d ai = %d stride =%d\n",swap_index,add_index,stride);
			}


			double tmp = local[swap_index];
			local[swap_index]  = local[add_index];
			local[add_index] += tmp;


		}	
		}

	__syncthreads();

	

	scanned[2*tid+ block_shift] = local[2*tid];
        scanned[(2*tid+1) + block_shift] =local[2*tid+1];

}

int main(int argc, char *argv[]){

	printf("START\n");
	if(true){

	unsigned int num_elements = THREADS * 4;
	size_t size = sizeof(double) * num_elements;
	double *h_scannable = (double *)malloc(size);
	double *h_prefixes   = (double *)malloc(size);

	double *d_scannable = NULL;
	double *d_prefixes = NULL;

	CHECK_CUDA(cudaMalloc((void **)&d_scannable, size));
	CHECK_CUDA(cudaMalloc((void **)&d_prefixes, size));

	for(int i = 0; i < num_elements;i++){
		h_scannable[i] = i;
		h_prefixes[i] = 0;
	}


	cudaMemcpy(d_scannable,h_scannable,size,cudaMemcpyHostToDevice);
	unsigned int block_size = THREADS;
	unsigned int num_blocks = (num_elements + block_size - 1) / block_size;
	printf("START KERNEL\n");
	belloch_scan_kernel<<<num_blocks,block_size>>>(d_scannable, d_prefixes,num_elements);
	cudaMemcpy(h_prefixes, d_prefixes,size,cudaMemcpyDeviceToHost);

	 for(int i = 0; i < num_elements;i++){
        	        
         	printf("index[%d] = %f\n",i,h_prefixes[i]);
      	}


	CHECK_CUDA(cudaFree(d_scannable));
	CHECK_CUDA(cudaFree(d_prefixes));
	
	free(h_scannable);
	free(h_prefixes);
	
	}
	else {

	printf("Hello World");
	}
}

