#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <cusparse.h>
#include "common.h"

const char* version_name = "cuSPARSE SpMV";\

#define CHECK_CUSPARSE(ret) if(ret != CUSPARSE_STATUS_SUCCESS) { fprintf(stderr, "error in line %d\n", __LINE__);}

typedef struct {
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void *dBuffer;
} additional_info_t;

typedef additional_info_t *info_ptr_t;

void preprocess(dist_matrix_t *mat, data_t *x, data_t *y) {
    info_ptr_t p = (info_ptr_t)malloc(sizeof(additional_info_t));
    cusparseCreate(&p->handle);
    cusparseCreateCsr(&p->matA, mat->global_m, mat->global_m, mat->global_nnz, mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&p->vecX, mat->global_m, x, CUDA_R_32F);
    cusparseCreateDnVec(&p->vecY, mat->global_m, y, CUDA_R_32F);
    size_t buffersize;
    data_t alpha, beta;
    alpha = 1.0;
    beta = 1.0;
    cusparseSpMV_bufferSize(p->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, p->matA, p->vecX, &beta, p->vecY, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &buffersize);
    p->dBuffer = NULL;
    cudaMalloc(&p->dBuffer, buffersize);
    //cusparseSetMatIndexBase(p->descrA, CUSPARSE_INDEX_BASE_ZERO);
    //cusparseSetMatType(p->descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    mat->additional_info = p;
}

void destroy_additional_info(void *additional_info) {
    info_ptr_t p = (info_ptr_t)additional_info;
    cusparseDestroySpMat(p->matA);
    cusparseDestroyDnVec(p->vecX);
    cusparseDestroyDnVec(p->vecY);
    cusparseDestroy(p->handle);
    cudaFree(p->dBuffer);
    free(p);
}

void spmv(dist_matrix_t *mat, const data_t* x, data_t* y) {
    int m = mat->global_m, nnz = mat->global_nnz;
    const data_t alpha = 1.0, beta = 1.0;
    info_ptr_t p = (info_ptr_t)mat->additional_info;

    cusparseSpMV(p->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, p->matA, p->vecX, &beta, p->vecY, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, p->dBuffer);
}
