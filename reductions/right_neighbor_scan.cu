
#include  "../common/cuda_helpers.c"

extern "C"{

#include "reduction_helpers.c"

}

__global__ void
vectorAdd(double *input,  double *output,int numElements)
{


	const int tid = threadIdx.x;
	const int idx = tid +  blockIdx.x * blockDim.x;
	double *offset = input + blockIdx.x * blockDim.x;

	if(idx >= numElements) return;


	for(int stride = 1; stride < blockDim.x; stride*=2){

		int index = 2*tid*stride;
		int neighbor = index + stride;
		if(neighbor < blockDim.x){
		offset[index] += offset[neighbor];
		}

		__syncthreads();
	}
	if(tid==0){output[blockIdx.x] = offset[0];}
}

void initialize_host(const size_t size,double **h_input,double **h_output){

	*h_input = (double *)malloc(size);
	*h_output = (double *) malloc(size);
}


void initialize_device(const size_t size,double **d_input,double **d_output){

	CHECK_CUDA(cudaMalloc((void **)d_output, size));
        CHECK_CUDA(cudaMalloc((void **)d_input, size));
}


void copy_host_to_device(const size_t size, double *h_input,double *h_output,double *d_input,double *d_output){

	CHECK_CUDA(cudaMemcpy(d_output, h_output, size, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
}

void copy_device_to_host(const size_t size, double *h_input,double *h_output,double *d_input,double *d_output){

        CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_input, d_input, size, cudaMemcpyDeviceToHost));
}



void destroy_host(double *h_input,double *h_output){

	free(h_input);
	free(h_output);
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
