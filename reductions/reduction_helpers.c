#include <stdio.h>

#define DIM 128

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



void check_reduction(double *input, double *sub_sums,const unsigned int num_elements, const unsigned int num_blocks){

	const int sum_first_n = (num_elements - 1)*(num_elements) /2 ;
	double cpu_sum = 0;
	double gpu_sum = 0;
	for(int i = 0; i < num_elements; i++){
		//cpu_sum += input[i];
		cpu_sum += (double)i;
		if (i < num_blocks)
		{
			gpu_sum+=sub_sums[i];
		}
	}


	if(fabs(gpu_sum - cpu_sum) > 1e-5){
		printf("REDUCTION FAILED gpu_sum = %f - cpu_sum = %f - sum first n = %d\n",gpu_sum,cpu_sum,sum_first_n);
	}
	else{
		printf("REDUCTION PASS gpu_sum = %f - cpu_sum = %f\n",gpu_sum,cpu_sum);

	}

}
