#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double data_t;
typedef int index_t;

char *cancat_name(const char *a, const char *b) {
    int l1 = strlen(a) - 4, l2 = strlen(b);
    char *c = (char*)malloc(sizeof(char) * (l1 + l2 + 1));
    if(c != NULL) {
        memcpy(c, a, l1 * sizeof(char));
        memcpy(c + l1, b, l2 * sizeof(char));
        c[l1 + l2] = '\0';
    }
    return c;
}

int my_m;

int process_mat(char *file_path, char *out_path)
{
    int global_m, global_nnz;
    int *r_pos, *c_idx;
    double *values;
    float *f_values;
    int count;
    int i;

    FILE *op;
    op = fopen(file_path, "rb");
    
    count = fread(&global_m, sizeof(index_t), 1, op);
    my_m = global_m;

    r_pos = (index_t*)malloc(sizeof(index_t) * (global_m + 1));

    count = fread(r_pos, sizeof(index_t), global_m + 1, op);
    global_nnz = r_pos[global_m];

    c_idx = (index_t*)malloc(sizeof(index_t) * global_nnz);
    values = (data_t*)malloc(sizeof(data_t) * global_nnz);

    count = fread(c_idx, sizeof(index_t), global_nnz, op);
    count = fread(values, sizeof(data_t), global_nnz, op);

    fclose(op);

    f_values = (float*)malloc(sizeof(float) * global_nnz);
    for (i = 0; i < global_m; i++) {
        f_values[i] = (float)values[i];
    }
    
    op = fopen(out_path, "wb");
    fwrite(&global_m, sizeof(int), 1, op);
    fwrite(r_pos, sizeof(int), global_m+1, op);
    fwrite(c_idx, sizeof(int), global_nnz, op);
    fwrite(f_values, sizeof(float), global_nnz, op);
    fclose(op);

    my_m = global_m;
    free(f_values);
    free(r_pos);
    free(c_idx);
    free(values);
}

int process_vector(char *file_path, char *out_path, int sz)
{
    double *val;
    float *f_val;
    int i;
    
    val = (double*)malloc(sizeof(double) * sz);
    f_val = (float*)malloc(sizeof(float) * sz);

    FILE *op;
    op = fopen(file_path, "rb");
    fread(val, sizeof(double), sz, op);
    fclose(op);

    for (i = 0; i < sz; i++) {
        f_val[i] = (float)val[i];
    }
    op = fopen(out_path, "wb");
    fwrite(f_val, sizeof(float), sz, op);
    fclose(op);
}

int main(int argc, char **argv)
{
    process_mat(argv[1], argv[2]);
    process_vector(cancat_name(argv[1], "_x.vec"), cancat_name(argv[2], "_x.vec"), my_m);
    process_vector(cancat_name(argv[1], "_y.vec"), cancat_name(argv[2], "_y.vec"), my_m*2);
}