#include <stdio.h>

#define DIM 128

void initialize_host(const size_t size,double **h_a_input,double **h_b_input,double **h_output){

	*h_a_input = (double *) malloc(size);
        *h_b_input = (double *)malloc(size);
        *h_output = (double *) malloc(size);
}


void initialize_device(const size_t size,double **d_a_input,double **d_b_input,double **d_output){

        CHECK_CUDA(cudaMalloc((void **)d_a_input, size));
	CHECK_CUDA(cudaMalloc((void **)d_b_input, size));
        CHECK_CUDA(cudaMalloc((void **)d_output, size));
}


void destroy_device(double *d_a_input,double *d_b_input,double *d_output){

	CHECK_CUDA(cudaFree(d_a_input));
	CHECK_CUDA(cudaFree(d_b_input));
	CHECK_CUDA(cudaFree(d_output));
}


void copy_host_to_device(const size_t size, double *h_a_input,double *h_b_input,double *h_output,double *d_a_input,double *d_b_input ,double *d_output){


	CHECK_CUDA(cudaMemcpy(d_a_input, h_a_input, size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b_input, h_b_input, size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_output, h_output, size, cudaMemcpyHostToDevice));
}

void copy_device_to_host(const size_t size, double *h_a_input,double *h_b_input,double *h_output,double *d_a_input,double *d_b_input,double *d_output){

        CHECK_CUDA(cudaMemcpy(h_a_input, d_a_input, size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_b_input, d_b_input, size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
}



void destroy_host(double *h_a_input,double *h_b_input,double *h_output){

        free(h_output);
        free(h_a_input);
        free(h_b_input);
}



void check_sum(double *h_a_input,double *h_b_input, double *h_output,const unsigned int num_elements, const unsigned int num_blocks){

	double gpu_sum;
	double cpu_sum;

	for(int i = 0; i < num_elements; i++)
	{
		gpu_sum = h_output[i];
		cpu_sum =  h_a_input[i] + h_b_input[i];
		if(fabs(gpu_sum - cpu_sum) > 1e-5)
		{
			printf("REDUCTION FAILED gpu_sum = %f - cpu_sum = %f\n",gpu_sum,cpu_sum);
		}
	}
}


void check_binary_op(double *h_a_input,double *h_b_input, double *h_output,const unsigned int num_elements, const unsigned int num_blocks){

	double gpu_sum;
        double cpu_sum;

	 for(int i = 0; i < num_elements; i++){
		      
		gpu_sum = h_output[i];
	 	cpu_sum =  max(h_a_input[i], h_b_input[i]);
	
		if(fabs(gpu_sum - cpu_sum) > 1e-5 ){printf("REDUCTION FAILED gpu_sum = %f - cpu_sum = %f\n",gpu_sum,cpu_sum); }

	}


}

