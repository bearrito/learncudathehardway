
#include  "../common/cuda_helpers.c"

extern "C"{

#include "sum_helpers.c"

}


typedef struct {
	int num_cols;
	int num_rows;
	// row major order
	double *elements;
} Matrix;



__global__ void matrix_multiply_kernel(Matrix a, Matrix b, Matrix c)
{
	if(a.num_cols != b.num_rows){
		printf("DIMENSION MISMATCH");
		return;
	}

	int tid = threadIdx.x;
	int index = tid + blockDim.x * blockIdx.x;

	if (tid >= a.num_rows) return;
        int a_row_offset = index*a.num_cols;
	int b_row_offset = index*b.num_cols;
	for(int outer_loop = 0; outer_loop < b.num_cols; outer_loop++)
	{	
		for(int inner_loop = 0;inner_loop < a.num_cols;inner_loop++){	

			int a_index = a_row_offset + inner_loop;
			int b_index = inner_loop*b.num_cols+outer_loop;
			int c_index = b_row_offset + outer_loop;
		
			//double addend =b.elements[bindex];
			double addend =a.elements[a_index]*b.elements[b_index];
			c.elements[c_index] += addend;
			if(index== 0){

				printf("outer=%d,inner=%d,cindex=%d,addend=%f\n",outer_loop,inner_loop,c_index,c.elements[c_index]);
			}			
		}

	}	


}


int indx(int row, int col,int num_cols){

	return num_cols*row + col;

}

void host_matrix_multiply(double *a , double *b ,double *c, int a_num_rows, int a_num_cols, int b_num_cols){

	for(int i = 0; i < a_num_rows; i++){
		for(int k = 0; k < b_num_cols;k++){
			double addend =0 ;
			double gpu_entry = c[indx(i,k,b_num_cols)];
			for(int j = 0; j < a_num_cols; j++){
			
				addend +=  a[indx(i,j,a_num_cols)] * b[indx(j,k,b_num_cols)];
			}
			if( fabs(gpu_entry -addend ) > 1e-5 )
			{

				printf("i=%d , j=%d , addend=%f, gpu=%f\n",i,k,addend,gpu_entry);
			}
			


		}
	}
}


 Matrix  create_matrix(int num_rows,int num_cols,double *h_elements){

	Matrix matrix;
	matrix.num_rows = num_rows;
	matrix.num_cols = num_cols;
	size_t size = num_rows * num_cols * sizeof(double);
	cudaMalloc(&matrix.elements,size);
	cudaMemcpy(matrix.elements,h_elements, size,cudaMemcpyHostToDevice);
	
	return matrix;


}

void destroy_matrix(Matrix matrix){
	
	cudaFree(matrix.elements);

}

int
main(void)
{
	int dim = 16;
	unsigned const int num_elements = 16 * 16;
	const size_t size = num_elements*sizeof(double);

	double *h_a_orig_input;
	double *h_b_orig_input;
	double *h_output;
	double *h_b_input;
	double *h_a_input;

	Matrix a;
	Matrix b;
	Matrix c;

	initialize_host(size,&h_a_input,&h_b_input,&h_output);

	  for(int i = 0; i < num_elements;i++){
                h_a_input[i] = (double)(rand()%11);
		h_b_input[i] = (double)(rand()%11);
                h_output[i] = 0;
        }

	a = create_matrix(dim,dim,h_a_input);
	b = create_matrix(dim,dim,h_b_input);
	c = create_matrix(dim,dim,h_output);

	
	h_a_orig_input = (double *)malloc(size);
	h_b_orig_input = (double *)malloc(size);

	memcpy(h_a_orig_input,h_a_input,size);
	memcpy(h_b_orig_input,h_b_input,size);
	
	int threadsPerBlock = 32;
	int blocksPerGrid =(num_elements + threadsPerBlock - 1) / threadsPerBlock;
    	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    	matrix_multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(a,b,c);
	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaMemcpy(h_output,c.elements,size,cudaMemcpyDeviceToHost));
	for(int i = 0;i < num_elements;i++){
		printf("index:%d=%f\n",i,h_output[i]);
		

	}
	
	host_matrix_multiply(h_a_orig_input,h_b_orig_input,h_output,dim,dim,dim);
	
	printf("DESTROYING\n");	
	destroy_matrix(a);
	destroy_matrix(b);
	destroy_matrix(c);
	destroy_host(h_a_input,h_b_input,h_output);
	free(h_a_orig_input);
	free(h_b_orig_input);
}
