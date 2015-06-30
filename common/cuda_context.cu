#include <stdlib.h>
#include "../common/cuda_helpers.c"

struct ScanContext{
	float *h_scannable;
	float *h_scanned;	
	float *d_scannable;
	float *d_scanned;
	size_t size;
};

ScanContext *create_scan_context(const int numElements){

	size_t size = sizeof(float) * numElements;
	struct ScanContext *ctxt = malloc(sizeof(struct ScanContext));
	ctxt->h_scannable=(float*)malloc(size);
	ctxt->h_scanned=(float*)malloc(size);
	float *d_scannable = NULL;
	float *d_scanned   = NULL;
	CHECK_CUDA(cudaMalloc((void **)&d_scannable, size));
	CHECK_CUDA(cudaMalloc((void **)&d_scanned, size));
	ctxt->size = size;
	return ctxt;
}


void copy_context_to_device(struct ScanContext *ctxt){

	CHECK_CUDA(cudaMemcpy(ctxt->d_scannable, ctxt->h_scannable,ctxt->size,cudaMemcpyHostToDevice));

}

void copy_device_to_host(struct ScanContext *ctxt){

        CHECK_CUDA(cudaMemcpy(ctxt->h_scannable, ctxt->d_scannable,ctxt->size,cudaMemcpyDeviceToHost));

}

void drestroy_scancontext(struct ScanContext *ctxt){


	free(ctxt->h_scannable);
	free(ctxt->h_scanned);

	CHECK_CUDA(cudaFree(ctxt->d_scanned));
	CHECK_CUDA(cudaFree(ctxt->d_scanned));
}
