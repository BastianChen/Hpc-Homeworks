#include <mmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>
#include <string.h>
#include <stdio.h>

const char *dgemm_desc = "Simple blocked dgemm.";

#define min(a, b) (((a)<(b))?(a):(b))
#define ARRAY(A, i, j) (A)[(j)*lda + (i)]

//2*2block
static void block2x2(int lda, int K, double *a, double *b, double *c) {
    register __m128d a0x_1x,
            bx0, bx1,
            c00_10,
            c01_11;

    double *c01_11_ptr = c + lda;

    c00_10 = _mm_loadu_pd(c);
    c01_11 = _mm_loadu_pd(c01_11_ptr);

    for (int x = 0; x < K; ++x) {
        a0x_1x = _mm_load_pd(a);
        a += 2;

        bx0 = _mm_loaddup_pd(b++);
        bx1 = _mm_loaddup_pd(b++);

        c00_10 = _mm_add_pd(c00_10, _mm_mul_pd(a0x_1x, bx0));
        c01_11 = _mm_add_pd(c01_11, _mm_mul_pd(a0x_1x, bx1));
    }

    _mm_storeu_pd(c, c00_10);
    _mm_storeu_pd(c01_11_ptr, c01_11);
}

//将A矩阵中的数据按行向量化
static inline void move_row2(int lda, const int K, double *a_src, double *a_dest) {
    for (int i = 0; i < K; ++i) {
        *a_dest++ = *a_src;
        *a_dest++ = *(a_src + 1);
        a_src += lda;
    }
}

//将B矩阵中的数据按列向量化
static inline void move_column2(int lda, const int K, double *b_src, double *b_dest) {
    double *b_ptr0, *b_ptr1;
    b_ptr0 = b_src;
    b_ptr1 = b_ptr0 + lda;

    for (int i = 0; i < K; ++i) {
        *b_dest++ = *b_ptr0++;
        *b_dest++ = *b_ptr1++;
    }
}

//3*3block
static void block3x3(int lda, int K, double *a, double *b, double *c) {
    register __m256d a0x_2x,
            bx0, bx1, bx2,
            c00_20, c01_21, c02_22;

    double *c01_21_ptr = c + lda;
    double *c02_22_ptr = c01_21_ptr + lda;

    c00_20 = _mm256_loadu_pd(c);
    c01_21 = _mm256_loadu_pd(c01_21_ptr);
    c02_22 = _mm256_loadu_pd(c02_22_ptr);

    for (int x = 0; x < K; ++x) {
        a0x_2x = _mm256_loadu_pd(a);
        a += 3;

        bx0 = _mm256_broadcast_sd(b++);
        bx1 = _mm256_broadcast_sd(b++);
        bx2 = _mm256_broadcast_sd(b++);

        c00_20 = _mm256_add_pd(c00_20, _mm256_mul_pd(a0x_2x, bx0));
        c01_21 = _mm256_add_pd(c01_21, _mm256_mul_pd(a0x_2x, bx1));
        c02_22 = _mm256_add_pd(c02_22, _mm256_mul_pd(a0x_2x, bx2));
    }

    _mm256_storeu_pd(c, c00_20);
    _mm256_storeu_pd(c01_21_ptr, c01_21);
    _mm256_storeu_pd(c02_22_ptr, c02_22);
}

static inline void move_row3(int lda, const int K, double *a_src, double *a_dest) {
    for (int i = 0; i < K; ++i) {
        *a_dest++ = *a_src;
        *a_dest++ = *(a_src + 1);
        *a_dest++ = *(a_src + 2);
        a_src += lda;
    }
}

static inline void move_column3(int lda, const int K, double *b_src, double *b_dest) {
    double *b_ptr0, *b_ptr1, *b_ptr2;
    b_ptr0 = b_src;
    b_ptr1 = b_ptr0 + lda;
    b_ptr2 = b_ptr1 + lda;

    for (int i = 0; i < K; ++i) {
        *b_dest++ = *b_ptr0++;
        *b_dest++ = *b_ptr1++;
        *b_dest++ = *b_ptr2++;
    }
}

//4*4block
static void block4x4(int lda, int K, double *a, double *b, double *c) {
    register __m256d a0x_3x,
            bx0, bx1, bx2, bx3,
            c00_30, c01_31,
            c02_32, c03_33;

    double *c01_31_ptr = c + lda;
    double *c02_32_ptr = c01_31_ptr + lda;
    double *c03_33_ptr = c02_32_ptr + lda;

    c00_30 = _mm256_loadu_pd(c);
    c01_31 = _mm256_loadu_pd(c01_31_ptr);
    c02_32 = _mm256_loadu_pd(c02_32_ptr);
    c03_33 = _mm256_loadu_pd(c03_33_ptr);

    for (int x = 0; x < K; ++x) {
        a0x_3x = _mm256_loadu_pd(a);
        a += 4;

        bx0 = _mm256_broadcast_sd(b++);
        bx1 = _mm256_broadcast_sd(b++);
        bx2 = _mm256_broadcast_sd(b++);
        bx3 = _mm256_broadcast_sd(b++);

        c00_30 = _mm256_add_pd(c00_30, _mm256_mul_pd(a0x_3x, bx0));
        c01_31 = _mm256_add_pd(c01_31, _mm256_mul_pd(a0x_3x, bx1));
        c02_32 = _mm256_add_pd(c02_32, _mm256_mul_pd(a0x_3x, bx2));
        c03_33 = _mm256_add_pd(c03_33, _mm256_mul_pd(a0x_3x, bx3));
    }

    _mm256_storeu_pd(c, c00_30);
    _mm256_storeu_pd(c01_31_ptr, c01_31);
    _mm256_storeu_pd(c02_32_ptr, c02_32);
    _mm256_storeu_pd(c03_33_ptr, c03_33);
}

//将A矩阵中的数据按行向量化
static inline void move_row4(int lda, const int K, double *a_src, double *a_dest) {
    for (int i = 0; i < K; ++i) {
        *a_dest++ = *a_src;
        *a_dest++ = *(a_src + 1);
        *a_dest++ = *(a_src + 2);
        *a_dest++ = *(a_src + 3);
        a_src += lda;
    }
}

//将B矩阵中的数据按列向量化
static inline void move_column4(int lda, const int K, double *b_src, double *b_dest) {
    double *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3;
    b_ptr0 = b_src;
    b_ptr1 = b_ptr0 + lda;
    b_ptr2 = b_ptr1 + lda;
    b_ptr3 = b_ptr2 + lda;

    for (int i = 0; i < K; ++i) {
        *b_dest++ = *b_ptr0++;
        *b_dest++ = *b_ptr1++;
        *b_dest++ = *b_ptr2++;
        *b_dest++ = *b_ptr3++;
    }
}

//6*6block
static void block6x6(int lda, int K, double *a, double *b, double *c) {
    register __m256d a0x_2x, a3x_5x,
            bx0, bx1, bx2, bx3, bx4, bx5,
            c00_20, c01_21, c02_22, c03_23, c04_24, c05_25,
            c30_50, c31_51, c32_52, c33_53, c34_54, c35_55;

    double *c01_21_ptr = c + lda;
    double *c02_22_ptr = c01_21_ptr + lda;
    double *c03_23_ptr = c02_22_ptr + lda;
    double *c04_24_ptr = c03_23_ptr + lda;
    double *c05_25_ptr = c04_24_ptr + lda;

    c00_20 = _mm256_loadu_pd(c);
    c01_21 = _mm256_loadu_pd(c01_21_ptr);
    c02_22 = _mm256_loadu_pd(c02_22_ptr);
    c03_23 = _mm256_loadu_pd(c03_23_ptr);
    c04_24 = _mm256_loadu_pd(c04_24_ptr);
    c05_25 = _mm256_loadu_pd(c05_25_ptr);

    c30_50 = _mm256_loadu_pd(c + 3);
    c31_51 = _mm256_loadu_pd(c01_21_ptr + 3);
    c32_52 = _mm256_loadu_pd(c02_22_ptr + 3);
    c33_53 = _mm256_loadu_pd(c03_23_ptr + 3);
    c34_54 = _mm256_loadu_pd(c04_24_ptr + 3);
    c35_55 = _mm256_loadu_pd(c05_25_ptr + 3);

    for (int x = 0; x < K; ++x) {
        a0x_2x = _mm256_loadu_pd(a);
        a3x_5x = _mm256_loadu_pd(a + 3);
        a += 6;

        bx0 = _mm256_broadcast_sd(b++);
        bx1 = _mm256_broadcast_sd(b++);
        bx2 = _mm256_broadcast_sd(b++);
        bx3 = _mm256_broadcast_sd(b++);
        bx4 = _mm256_broadcast_sd(b++);
        bx5 = _mm256_broadcast_sd(b++);

        c00_20 = _mm256_add_pd(c00_20, _mm256_mul_pd(a0x_2x, bx0));
        c01_21 = _mm256_add_pd(c01_21, _mm256_mul_pd(a0x_2x, bx1));
        c02_22 = _mm256_add_pd(c02_22, _mm256_mul_pd(a0x_2x, bx2));
        c03_23 = _mm256_add_pd(c03_23, _mm256_mul_pd(a0x_2x, bx3));
        c04_24 = _mm256_add_pd(c04_24, _mm256_mul_pd(a0x_2x, bx4));
        c05_25 = _mm256_add_pd(c05_25, _mm256_mul_pd(a0x_2x, bx5));

        c30_50 = _mm256_add_pd(c30_50, _mm256_mul_pd(a3x_5x, bx0));
        c31_51 = _mm256_add_pd(c31_51, _mm256_mul_pd(a3x_5x, bx1));
        c32_52 = _mm256_add_pd(c32_52, _mm256_mul_pd(a3x_5x, bx2));
        c33_53 = _mm256_add_pd(c33_53, _mm256_mul_pd(a3x_5x, bx3));
        c34_54 = _mm256_add_pd(c34_54, _mm256_mul_pd(a3x_5x, bx4));
        c35_55 = _mm256_add_pd(c35_55, _mm256_mul_pd(a3x_5x, bx5));
    }

    _mm256_storeu_pd(c, c00_20);
    _mm256_storeu_pd(c01_21_ptr, c01_21);
    _mm256_storeu_pd(c02_22_ptr, c02_22);
    _mm256_storeu_pd(c03_23_ptr, c03_23);
    _mm256_storeu_pd(c04_24_ptr, c04_24);
    _mm256_storeu_pd(c05_25_ptr, c05_25);

    _mm256_storeu_pd(c + 3, c30_50);
    _mm256_storeu_pd(c01_21_ptr + 3, c31_51);
    _mm256_storeu_pd(c02_22_ptr + 3, c32_52);
    _mm256_storeu_pd(c03_23_ptr + 3, c33_53);
    _mm256_storeu_pd(c04_24_ptr + 3, c34_54);
    _mm256_storeu_pd(c05_25_ptr + 3, c35_55);
}

static inline void move_row6(int lda, const int K, double *a_src, double *a_dest) {
    for (int i = 0; i < K; ++i) {
        *a_dest++ = *a_src;
        *a_dest++ = *(a_src + 1);
        *a_dest++ = *(a_src + 2);
        *a_dest++ = *(a_src + 3);
        *a_dest++ = *(a_src + 4);
        *a_dest++ = *(a_src + 5);
        a_src += lda;
    }
}

static inline void move_column6(int lda, const int K, double *b_src, double *b_dest) {
    double *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3, *b_ptr4, *b_ptr5;
    b_ptr0 = b_src;
    b_ptr1 = b_ptr0 + lda;
    b_ptr2 = b_ptr1 + lda;
    b_ptr3 = b_ptr2 + lda;
    b_ptr4 = b_ptr3 + lda;
    b_ptr5 = b_ptr4 + lda;

    for (int i = 0; i < K; ++i) {
        *b_dest++ = *b_ptr0++;
        *b_dest++ = *b_ptr1++;
        *b_dest++ = *b_ptr2++;
        *b_dest++ = *b_ptr3++;
        *b_dest++ = *b_ptr4++;
        *b_dest++ = *b_ptr5++;
    }
}

//8*8block
static void block8x8(int lda, int K, double *a, double *b, double *c) {
    register __m256d a0x_3x, a4x_7x,
            bx0, bx1, bx2, bx3,
            bx4, bx5, bx6, bx7,
            c00_30, c40_70,
            c01_31, c41_71,
            c02_32, c42_72,
            c03_33, c43_73,
            c04_34, c44_74,
            c05_35, c45_75,
            c06_36, c46_76,
            c07_37, c47_77;

    double *c01_31_ptr = c + lda;
    double *c02_32_ptr = c01_31_ptr + lda;
    double *c03_33_ptr = c02_32_ptr + lda;
    double *c04_34_ptr = c03_33_ptr + lda;
    double *c05_35_ptr = c04_34_ptr + lda;
    double *c06_36_ptr = c05_35_ptr + lda;
    double *c07_37_ptr = c06_36_ptr + lda;

    c00_30 = _mm256_loadu_pd(c);
    c40_70 = _mm256_loadu_pd(c + 4);
    c01_31 = _mm256_loadu_pd(c01_31_ptr);
    c41_71 = _mm256_loadu_pd(c01_31_ptr + 4);
    c02_32 = _mm256_loadu_pd(c02_32_ptr);
    c42_72 = _mm256_loadu_pd(c02_32_ptr + 4);
    c03_33 = _mm256_loadu_pd(c03_33_ptr);
    c43_73 = _mm256_loadu_pd(c03_33_ptr + 4);
    c04_34 = _mm256_loadu_pd(c04_34_ptr);
    c44_74 = _mm256_loadu_pd(c04_34_ptr + 4);
    c05_35 = _mm256_loadu_pd(c05_35_ptr);
    c45_75 = _mm256_loadu_pd(c05_35_ptr + 4);
    c06_36 = _mm256_loadu_pd(c06_36_ptr);
    c46_76 = _mm256_loadu_pd(c06_36_ptr + 4);
    c07_37 = _mm256_loadu_pd(c07_37_ptr);
    c47_77 = _mm256_loadu_pd(c07_37_ptr + 4);

    for (int x = 0; x < K; ++x) {
        a0x_3x = _mm256_loadu_pd(a);
        a4x_7x = _mm256_loadu_pd(a + 4);
        a += 8;

        bx0 = _mm256_broadcast_sd(b++);
        bx1 = _mm256_broadcast_sd(b++);
        bx2 = _mm256_broadcast_sd(b++);
        bx3 = _mm256_broadcast_sd(b++);
        bx4 = _mm256_broadcast_sd(b++);
        bx5 = _mm256_broadcast_sd(b++);
        bx6 = _mm256_broadcast_sd(b++);
        bx7 = _mm256_broadcast_sd(b++);

        c00_30 = _mm256_add_pd(c00_30, _mm256_mul_pd(a0x_3x, bx0));
        c40_70 = _mm256_add_pd(c40_70, _mm256_mul_pd(a4x_7x, bx0));
        c01_31 = _mm256_add_pd(c01_31, _mm256_mul_pd(a0x_3x, bx1));
        c41_71 = _mm256_add_pd(c41_71, _mm256_mul_pd(a4x_7x, bx1));
        c02_32 = _mm256_add_pd(c02_32, _mm256_mul_pd(a0x_3x, bx2));
        c42_72 = _mm256_add_pd(c42_72, _mm256_mul_pd(a4x_7x, bx2));
        c03_33 = _mm256_add_pd(c03_33, _mm256_mul_pd(a0x_3x, bx3));
        c43_73 = _mm256_add_pd(c43_73, _mm256_mul_pd(a4x_7x, bx3));
        c04_34 = _mm256_add_pd(c04_34, _mm256_mul_pd(a0x_3x, bx4));
        c44_74 = _mm256_add_pd(c44_74, _mm256_mul_pd(a4x_7x, bx4));
        c05_35 = _mm256_add_pd(c05_35, _mm256_mul_pd(a0x_3x, bx5));
        c45_75 = _mm256_add_pd(c45_75, _mm256_mul_pd(a4x_7x, bx5));
        c06_36 = _mm256_add_pd(c06_36, _mm256_mul_pd(a0x_3x, bx6));
        c46_76 = _mm256_add_pd(c46_76, _mm256_mul_pd(a4x_7x, bx6));
        c07_37 = _mm256_add_pd(c07_37, _mm256_mul_pd(a0x_3x, bx7));
        c47_77 = _mm256_add_pd(c47_77, _mm256_mul_pd(a4x_7x, bx7));
    }

    _mm256_storeu_pd(c, c00_30);
    _mm256_storeu_pd(c + 4, c40_70);
    _mm256_storeu_pd(c01_31_ptr, c01_31);
    _mm256_storeu_pd(c01_31_ptr + 4, c41_71);
    _mm256_storeu_pd(c02_32_ptr, c02_32);
    _mm256_storeu_pd(c02_32_ptr + 4, c42_72);
    _mm256_storeu_pd(c03_33_ptr, c03_33);
    _mm256_storeu_pd(c03_33_ptr + 4, c43_73);
    _mm256_storeu_pd(c04_34_ptr, c04_34);
    _mm256_storeu_pd(c04_34_ptr + 4, c44_74);
    _mm256_storeu_pd(c05_35_ptr, c05_35);
    _mm256_storeu_pd(c05_35_ptr + 4, c45_75);
    _mm256_storeu_pd(c06_36_ptr, c06_36);
    _mm256_storeu_pd(c06_36_ptr + 4, c46_76);
    _mm256_storeu_pd(c07_37_ptr, c07_37);
    _mm256_storeu_pd(c07_37_ptr + 4, c47_77);
}

static inline void move_row8(int lda, const int K, double *a_src, double *a_dest) {
    for (int i = 0; i < K; ++i) {
        *a_dest++ = *a_src;
        *a_dest++ = *(a_src + 1);
        *a_dest++ = *(a_src + 2);
        *a_dest++ = *(a_src + 3);
        *a_dest++ = *(a_src + 4);
        *a_dest++ = *(a_src + 5);
        *a_dest++ = *(a_src + 6);
        *a_dest++ = *(a_src + 7);
        a_src += lda;
    }
}

static inline void move_column8(int lda, const int K, double *b_src, double *b_dest) {
    double *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3,
            *b_ptr4, *b_ptr5, *b_ptr6, *b_ptr7;
    b_ptr0 = b_src;
    b_ptr1 = b_ptr0 + lda;
    b_ptr2 = b_ptr1 + lda;
    b_ptr3 = b_ptr2 + lda;
    b_ptr4 = b_ptr3 + lda;
    b_ptr5 = b_ptr4 + lda;
    b_ptr6 = b_ptr5 + lda;
    b_ptr7 = b_ptr6 + lda;

    for (int i = 0; i < K; ++i) {
        *b_dest++ = *b_ptr0++;
        *b_dest++ = *b_ptr1++;
        *b_dest++ = *b_ptr2++;
        *b_dest++ = *b_ptr3++;
        *b_dest++ = *b_ptr4++;
        *b_dest++ = *b_ptr5++;
        *b_dest++ = *b_ptr6++;
        *b_dest++ = *b_ptr7++;
    }
}

static void inline

do_block(int block_size, int lda, int M, int N, int K, double *A, double *B, double *C) {
    double A_block[M * K], B_block[K * N];
    double *a_ptr, *b_ptr, *c;

    int column_max = N - block_size + 1;
    int row_max = M - block_size + 1;
    int flag_row = M % block_size;
    int flag_column = N % block_size;

    register int i = 0, j = 0, p = 0;
    // 先对数据进行向量化再进行分块计算
    for (j = 0; j < column_max; j += block_size) {
        b_ptr = &B_block[j * K];

        switch (block_size) {
            // 采用2*2的block
            case 2:
                move_column2(lda, K, B + j * lda, b_ptr);
                for (i = 0; i < row_max; i += block_size) {
                    a_ptr = &A_block[i * K];
                    if (j == 0) move_row2(lda, K, A + i, a_ptr);
                    c = C + i + j * lda;
                    block2x2(lda, K, a_ptr, b_ptr, c);
                }
                break;
                // 采用3*3的block
            case 3:
                move_column3(lda, K, B + j * lda, b_ptr);
                for (i = 0; i < row_max; i += block_size) {
                    a_ptr = &A_block[i * K];
                    if (j == 0) move_row3(lda, K, A + i, a_ptr);
                    c = C + i + j * lda;
                    block3x3(lda, K, a_ptr, b_ptr, c);
                }
                break;
                // 采用4*4的block
            case 4:
                move_column4(lda, K, B + j * lda, b_ptr);
                for (i = 0; i < row_max; i += block_size) {
                    a_ptr = &A_block[i * K];
                    if (j == 0) move_row4(lda, K, A + i, a_ptr);
                    c = C + i + j * lda;
                    block4x4(lda, K, a_ptr, b_ptr, c);
                }
                break;
                // 采用6*6的block
            case 6:
                move_column6(lda, K, B + j * lda, b_ptr);
                for (i = 0; i < row_max; i += block_size) {
                    a_ptr = &A_block[i * K];
                    if (j == 0) move_row6(lda, K, A + i, a_ptr);
                    c = C + i + j * lda;
                    block6x6(lda, K, a_ptr, b_ptr, c);
                }
                break;
                // 采用8*8的block
            case 8:
                move_column8(lda, K, B + j * lda, b_ptr);
                for (i = 0; i < row_max; i += block_size) {
                    a_ptr = &A_block[i * K];
                    if (j == 0) move_row8(lda, K, A + i, a_ptr);
                    c = C + i + j * lda;
                    block8x8(lda, K, a_ptr, b_ptr, c);
                }
                break;
            default:
                printf("Block size error!");
        }
    }
    //处理行未处理的数据
    if (flag_row != 0) {
        //遍历A矩阵的每行
        for (; i < M; ++i) {
            //遍历B矩阵的每列
            for (p = 0; p < N; ++p) {
                //计算C[i,j]
                register double c_ip = ARRAY(C, i, p);
                for (int k = 0; k < K; ++k)
                    c_ip += ARRAY(A, i, k) * ARRAY(B, k, p);
                ARRAY(C, i, p) = c_ip;
            }
        }
    }

    //处理列未处理的数据
    if (flag_column != 0) {
        row_max = M - flag_row;
        //遍历B矩阵的每列
        for (; j < N; ++j) {
            //遍历A矩阵的每行
            for (i = 0; i < row_max; ++i) {
                //计算C[i,j]
                register double cij = ARRAY(C, i, j);
                for (int k = 0; k < K; ++k)
                    cij += ARRAY(A, i, k) * ARRAY(B, k, j);
                ARRAY(C, i, j) = cij;
            }
        }
    }
}

void square_dgemm(int lda, double *A, double *B, double *C) {
    //定义两层分块
    register int BLOCK1 = 128;
    register int BLOCK2 = 256;
    register int block_size = 4;

    //最外层256*256大小的block
    for (int x = 0; x < lda; x += BLOCK2) {
        int lim_k = x + min (BLOCK2, lda - x);
        for (int y = 0; y < lda; y += BLOCK2) {
            int lim_j = y + min (BLOCK2, lda - y);
            for (int z = 0; z < lda; z += BLOCK2) {
                int lim_i = z + min (BLOCK2, lda - z);
                //第二层128*128大小的block
                for (int k = x; k < lim_k; k += BLOCK1) {
                    int K = min (BLOCK1, lim_k - k);
                    for (int j = y; j < lim_j; j += BLOCK1) {
                        int N = min (BLOCK1, lim_j - j);
                        for (int i = z; i < lim_i; i += BLOCK1) {
                            int M = min (BLOCK1, lim_i - i);
                            //最内层block_size*block_size大小的block
                            do_block(block_size, lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                        }
                    }
                }
            }
        }
    }
}
