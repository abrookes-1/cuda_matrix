#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "util.h"

/*
WRITE CUDA KERNEL FOR TRANSPOSE HERE
*/
const int CHUNK_SIZE = 32;
const int CHUNK_ROWS = 8;

__global__ void matrix_t(int* data, int* out, int* rows, int* cols){
    __shared__ int chunk[CHUNK_SIZE][CHUNK_SIZE];
    int x = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    int y = blockIdx.y * CHUNK_SIZE + threadIdx.y;

    for (int i=0; i<CHUNK_SIZE; i+= CHUNK_ROWS) {
        chunk[threadIdx.x][threadIdx.y+i] = data[(y + i) * *cols + x];
    }
    __syncthreads();

    x = blockIdx.y * CHUNK_SIZE + threadIdx.x;
    y = blockIdx.x * CHUNK_SIZE + threadIdx.y;

    for (int i=0; i<CHUNK_SIZE; i+= CHUNK_ROWS) {
        if (x < *rows && y+i < *cols) {
            out[(y + i) * *rows + x] = chunk[threadIdx.y + i][threadIdx.x];
//            out[(y + i) * *rows + x] = 1;
        }
    }
}

int * serial_implementation(int * data_in, int rows, int cols) {
    int * out = (int *)malloc(sizeof(int) * rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[j * rows + i] = data_in[i * cols + j];
        }
    }
    return out;
}

int main(int argc, char ** argv) {

    int rows = 0, cols = 0;

    assert(argc == 2);
    int * data = read_file(argv[1], &rows, &cols);

    cudaStream_t stream;
    cudaEvent_t begin, end;
    cudaStreamCreate(&stream);
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    int * transpose_h = (int *)malloc(sizeof(int) * rows * cols); // THIS VARIABLE SHOULD HOLD THE TOTAL COUNT BY THE END

    /*
    PERFORM NECESSARY VARIABLE DECLARATIONS HERE
    PERFORM NECESSARY DATA TRANSFER HERE
    */
    int *data_p, *rows_p, *cols_p;

    cudaMallocManaged(&data_p, rows * cols * sizeof(int));
    cudaMallocManaged(&transpose_h, rows * cols * sizeof(int));
    cudaMallocManaged(&rows_p, sizeof(int));
    cudaMallocManaged(&cols_p, sizeof(int));

    for (int i=0; i<rows*cols; i++){
        data_p[i] = data[i];
    }
    *rows_p = rows;
    *cols_p = cols;

//    for (int i=0; i<100; i++){
//        printf("%i", data_p[i]);
//    }
//    printf("\n");

    cudaEventRecord(begin, stream);

    /*
    LAUNCH KERNEL HERE
    */
    size_t thread_x = CHUNK_SIZE;
    size_t thread_y = CHUNK_SIZE;
    // ceiling of cols/threads_x
    size_t grid_x = (cols + thread_x - 1) / thread_x;
    // ceiling of rows/threads_y
    size_t grid_y = (rows + thread_y - 1) / thread_y;

//    dim3 block_dim(thread_x, thread_y, 1);
//    dim3 grid_dim(grid_x, grid_y, 1);
    dim3 grid_dim(grid_x, grid_y, 1);
    dim3 block_dim(CHUNK_SIZE, CHUNK_ROWS, 1);
//    printf("threads per block: %i, %i\n", block_dim.x, block_dim.y);
//    printf("blocks per grid: %i, %i\n", grid_dim.x, grid_dim.y);

    matrix_t <<<grid_dim, block_dim>>> (data_p, transpose_h, rows_p, cols_p);
    cudaDeviceSynchronize();

    cudaEventRecord(end, stream);

    /*
    PERFORM NECESSARY DATA TRANSFER HERE
    */

//    for (int i=0; i<(cols); i++){
//        for (int j=0; j<(rows); j++){
//            printf("%i_", transpose_h[i * (rows) + j]);
//        }
//        printf("\n");
//    }
//    for (int i=0; i<100; i++){
//        printf("%i_", transpose_h[i]);
//    }
//    printf("\n");

    cudaStreamSynchronize(stream);

    float ms;
    cudaEventElapsedTime(&ms, begin, end);
    printf("Elapsed time: %f ms\n", ms);

    /* 

    DEALLOCATE RESOURCES HERE

    */

    int * transpose_serial = serial_implementation(data, rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        if (transpose_h[i] != transpose_serial[i]) {
            printf("ERROR: %d != %d\n", transpose_serial[i], transpose_h[i]);
            printf("IDX: %i\n", i);
            exit(-1);
        }
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);

    free(data);
    free(transpose_serial);
    cudaFree(data);
    cudaFree(transpose_h);

    return 0;
}
