


typedef struct {
	int num_columns;
	int num_rows;
	// row major order
	double *elements;
} Matrix;



 Matrix  create_matrix(int num_rows,int num_cols,double *h_elements){

	Matrix matrix;
	matrix.num_rows = num_rows;
	matrix.num_columns = num_cols;
	size_t size = num_rows * num_cols * sizeof(double);
	cudaMalloc(&matrix.elements,size);
	cudaMemcpy(matrix.elements,h_elements, size,cudaMemcpyHostToDevice);
	
	return matrix;


}

void destroy_matrix(Matrix matrix){
	
	cudaFree(matrix.elements);

}
