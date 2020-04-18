#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "util.h"

/*
WRITE CUDA KERNEL FOR COUNT HERE
*/


int serial_implementation(int * data, int rows, int cols) {
    int count = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (data[i] == 1) count++;
    }
    return count;
}

__global__ void matrix_count(int* data, int* out, int* rows, int* cols){
    int id = threadIdx.x;
    if (id < *rows * *cols){
//        if (data[id] == 1)
//            out[id] = 1;
//        else
//            out[id] = 0;
        if (data[id] == 1)
            atomicAdd(out, 1);
    }
}

//__global__ void array_sum(int* arr, int* arr_len, int* width){
//    int id = threadIdx.x;
//    if (id < arr_len){
//
//    }
//}

int main(int argc, char ** argv) {
    
    int rows = 0, cols = 0;

    assert(argc == 2);
    int * data = read_file(argv[1], &rows, &cols);

    cudaStream_t stream;
    cudaEvent_t begin, end;
    cudaStreamCreate(&stream);
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    int *count_h = 0; // THIS VARIABLE SHOULD HOLD THE TOTAL COUNT BY THE END

    /*
    PERFORM NECESSARY VARIABLE DECLARATIONS HERE
    PERFORM NECESSARY DATA TRANSFER HERE
    */
    int *rows_p, *cols_p;
    int *data_p;

    cudaMallocManaged(&data_p, rows * cols * sizeof(int));
    cudaMallocManaged(&count_h, sizeof(int));
    cudaMallocManaged(&rows_p, sizeof(int));
    cudaMallocManaged(&cols_p, sizeof(int));

    *rows_p = rows;
    *cols_p = cols;
    *count_h = 0;
    for (int i=0; i<rows*cols; i++){
        data_p[i] = data[i];
    }

    cudaEventRecord(begin, stream);

    /*
    LAUNCH KERNEL HERE
    */

    matrix_count <<<1, rows*cols>>> (data_p, count_h, rows_p, cols_p);
    cudaDeviceSynchronize();

    cudaEventRecord(end, stream);
    /*
    PERFORM NECESSARY DATA TRANSFER HERE
    */

    cudaStreamSynchronize(stream);


    float ms;
    cudaEventElapsedTime(&ms, begin, end);
    printf("Elapsed time: %f ms\n", ms);

    /*
    DEALLOCATE RESOURCES HERE
    */
    int count_serial = serial_implementation(data, rows, cols);
    if (count_serial != *count_h) {
        printf("ERROR: %d != %d\n", count_serial, *count_h);
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);

    free(data);
    cudaFree(data);
    cudaFree(rows_p);
    cudaFree(cols_p);

    return 0;
}