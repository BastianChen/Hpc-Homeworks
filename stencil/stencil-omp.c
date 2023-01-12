#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include <omp.h>
#include <immintrin.h>

const char *version_name = "omp version";


void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    /* Naive implementation uses Process 0 to do all computations */
    if (grid_info->p_id == 0) {
        grid_info->local_size_x = grid_info->global_size_x;
        grid_info->local_size_y = grid_info->global_size_y;
        grid_info->local_size_z = grid_info->global_size_z;
    } else {
        grid_info->local_size_x = 0;
        grid_info->local_size_y = 0;
        grid_info->local_size_z = 0;
    }
    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = 0;
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

void cal7(cptr_t a0, ptr_t a1, cptr_t b0, ptr_t b1, cptr_t c0, ptr_t c1, int x, int y, int z, int ldx, int ldy) {
    __m256d a7 = _mm256_setzero_pd();
    __m256d b7 = _mm256_setzero_pd();
    __m256d c7 = _mm256_setzero_pd();
    __m256d in1, in2, in3, in4, in5, in6, in7;
    __m256d l1, l2, l3, l4, l5, l6, l7;

    in1 = _mm256_loadu_pd(a0 + INDEX(x, y, z, ldx, ldy));
    in2 = _mm256_loadu_pd(a0 + INDEX(x - 1, y, z, ldx, ldy));
    in3 = _mm256_loadu_pd(a0 + INDEX(x + 1, y, z, ldx, ldy));
    in4 = _mm256_loadu_pd(a0 + INDEX(x, y - 1, z, ldx, ldy));
    in5 = _mm256_loadu_pd(a0 + INDEX(x, y + 1, z, ldx, ldy));
    in6 = _mm256_loadu_pd(a0 + INDEX(x, y, z - 1, ldx, ldy));
    in7 = _mm256_loadu_pd(a0 + INDEX(x, y, z + 1, ldx, ldy));
    a7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_ZZZ), in1, a7);
    a7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_NZZ), in2, a7);
    a7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_PZZ), in3, a7);
    a7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_ZNZ), in4, a7);
    a7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_ZPZ), in5, a7);
    a7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_ZZN), in6, a7);
    a7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_ZZP), in7, a7);

    in1 = _mm256_loadu_pd(b0 + INDEX(x, y, z, ldx, ldy));
    in2 = _mm256_loadu_pd(b0 + INDEX(x - 1, y, z, ldx, ldy));
    in3 = _mm256_loadu_pd(b0 + INDEX(x + 1, y, z, ldx, ldy));
    in4 = _mm256_loadu_pd(b0 + INDEX(x, y - 1, z, ldx, ldy));
    in5 = _mm256_loadu_pd(b0 + INDEX(x, y + 1, z, ldx, ldy));
    in6 = _mm256_loadu_pd(b0 + INDEX(x, y, z - 1, ldx, ldy));
    in7 = _mm256_loadu_pd(b0 + INDEX(x, y, z + 1, ldx, ldy));
    b7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_PNZ), in1, b7);
    b7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_NPZ), in2, b7);
    b7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_PPZ), in3, b7);
    b7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_NZN), in4, b7);
    b7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_PZN), in5, b7);
    b7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_PZP), in6, b7);
    b7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_NZP), in7, b7);

    in1 = _mm256_loadu_pd(c0 + INDEX(x, y, z, ldx, ldy));
    in2 = _mm256_loadu_pd(c0 + INDEX(x - 1, y, z, ldx, ldy));
    in3 = _mm256_loadu_pd(c0 + INDEX(x + 1, y, z, ldx, ldy));
    in4 = _mm256_loadu_pd(c0 + INDEX(x, y - 1, z, ldx, ldy));
    in5 = _mm256_loadu_pd(c0 + INDEX(x, y + 1, z, ldx, ldy));
    in6 = _mm256_loadu_pd(c0 + INDEX(x, y, z - 1, ldx, ldy));
    in7 = _mm256_loadu_pd(c0 + INDEX(x, y, z + 1, ldx, ldy));
    c7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_PNN), in1, c7);
    c7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_PPN), in2, c7);
    c7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_PPN), in3, c7);
    c7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_NNP), in4, c7);
    c7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_PNP), in5, c7);
    c7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_NPP), in6, c7);
    c7 = _mm256_fmadd_pd(_mm256_set1_pd((double) ALPHA_PPP), in7, c7);

    _mm256_storeu_pd(a1 + INDEX(x, y, z, ldx, ldy), a7 + (b7 * c7) / (b7 + c7));
    _mm256_storeu_pd(b1 + INDEX(x, y, z, ldx, ldy), b7 + (a7 * c7) / (a7 + c7));
    _mm256_storeu_pd(c1 + INDEX(x, y, z, ldx, ldy), c7 + (a7 * b7) / (a7 + b7));
}

ptr_t stencil_7(ptr_t A0, ptr_t A1, ptr_t B0, ptr_t B1, ptr_t C0, ptr_t C1, const dist_grid_info_t *grid_info, int nt) {
    ptr_t bufferx[2] = {A0, A1};
    ptr_t buffery[2] = {B0, B1};
    ptr_t bufferz[2] = {C0, C1};

    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    omp_set_num_threads(1);

    for (int t = 0; t < nt; ++t) {
        cptr_t a0 = bufferx[t % 2];
        ptr_t a1 = bufferx[(t + 1) % 2];

        cptr_t b0 = buffery[t % 2];
        ptr_t b1 = buffery[(t + 1) % 2];

        cptr_t c0 = bufferz[t % 2];
        ptr_t c1 = bufferz[(t + 1) % 2];

        #pragma omp parallel for
        for (int z = z_start; z < z_end; ++z) {
            for (int y = y_start; y < y_end; ++y) {
                for (int x = x_start; x < x_end; x += 4) {
                    cal7(a0, a1, b0, b1, c0, c1, x, y, z, ldx, ldy);
                }
            }
        }
    }
    return bufferx[nt % 2];
}


