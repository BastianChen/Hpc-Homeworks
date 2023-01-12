#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include "common.h"
#include "utils.h"

const char *version_name = "optimized version";

void preprocess(dist_matrix_t *mat, data_t *x, data_t *y) {
}

void destroy_additional_info(void *additional_info) {
}

// 在寄存器之间进行数据交换，比通过共享内存更有效，共享内存需要加载、存储和额外的寄存器来保存地址。
template<unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0<-16, 1<-17, 2<-18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0<-8, 1<-9, 2<-10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0<-4, 1<-5, 2<-6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0<-2, 1<-3, 4<-6, 5<-7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0<-1, 2<-3, 4<-5, etc.
    return sum;
}

template<unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__global__ void
spmv_optimized_kernel(int row_num, const index_t *r_pos, const index_t *c_idx, const data_t *values,
                      const data_t *x, data_t *y) {
    const index_t THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
    const index_t thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const index_t thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const index_t row_id = thread_id / THREADS_PER_VECTOR;               // global vector index

    if (row_id < row_num) {
        const index_t row_start = r_pos[row_id];
        const index_t row_end = r_pos[row_id + 1];

        // initialize local sum
        data_t sum = 0;

        // accumulate local sums
        #pragma unroll
        for (index_t i = row_start + thread_lane; i < row_end; i += THREADS_PER_VECTOR)
            sum += values[i] * x[c_idx[i]];

        sum = warpReduceSum<THREADS_PER_VECTOR>(sum);
        if (thread_lane == 0) {
            y[row_id] = sum;
        }
    }
}


void spmv(dist_matrix_t *mat, const data_t *vector, data_t *out) {
    // 32 thread for a row
    int row_num = mat->global_m;
    int mean_col_num = (mat->global_nnz) / row_num;

    // 根据不同数据的情况调整threads跟blocks
    if (mean_col_num <= 2) {
        const int THREADS_PER_VECTOR = 2;
        const unsigned int VECTORS_PER_BLOCK = 128;
        const unsigned int NUM_BLOCKS = static_cast<unsigned int>((row_num + (VECTORS_PER_BLOCK - 1)) /
                                                                  VECTORS_PER_BLOCK);
        spmv_optimized_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, 256>>>
                (row_num, mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, vector, out);
    } else if (mean_col_num > 2 && mean_col_num <= 4) {
        const int THREADS_PER_VECTOR = 4;
        const unsigned int VECTORS_PER_BLOCK = 64;
        const unsigned int NUM_BLOCKS = static_cast<unsigned int>((row_num + (VECTORS_PER_BLOCK - 1)) /
                                                                  VECTORS_PER_BLOCK);
        spmv_optimized_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, 256>>>
                (row_num, mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, vector, out);
    } else if (mean_col_num > 4 && mean_col_num <= 8) {
        const int THREADS_PER_VECTOR = 8;
        const unsigned int VECTORS_PER_BLOCK = 32;
        const unsigned int NUM_BLOCKS = static_cast<unsigned int>((row_num + (VECTORS_PER_BLOCK - 1)) /
                                                                  VECTORS_PER_BLOCK);
        spmv_optimized_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, 256>>>
                (row_num, mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, vector, out);
    } else if (mean_col_num > 8 && mean_col_num <= 16) {
        const int THREADS_PER_VECTOR = 16;
        const unsigned int VECTORS_PER_BLOCK = 16;
        const unsigned int NUM_BLOCKS = static_cast<unsigned int>((row_num + (VECTORS_PER_BLOCK - 1)) /
                                                                  VECTORS_PER_BLOCK);
        spmv_optimized_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, 256>>>
                (row_num, mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, vector, out);
    } else if (mean_col_num > 16) {
        const int THREADS_PER_VECTOR = 32;
        const unsigned int VECTORS_PER_BLOCK = 8;
        const unsigned int NUM_BLOCKS = static_cast<unsigned int>((row_num + (VECTORS_PER_BLOCK - 1)) /
                                                                  VECTORS_PER_BLOCK);
        spmv_optimized_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, 256>>>
                (row_num, mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, vector, out);
    }
}
