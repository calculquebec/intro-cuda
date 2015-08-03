/**
 * This program is designed to produce output 'data = 7'.
 * However, errors have been intentionally placed in the program
 * as an error checking exercise.
 */
#include <stdio.h>
#include <stdlib.h>

__global__ void setData(int *ptr) {
    *ptr = 7;
}

int main(int, char**) {
    int *data_d = 0;
    int *data_h = 0;

    cudaMalloc((void**)&data_d, UINT_MAX*sizeof(int));
    data_h = (int *)malloc(sizeof(int));

    setData<<<1,1>>>(0);
    cudaMemcpy(data_h, data_d, sizeof(int), cudaMemcpyDeviceToHost);

    printf("data = %d\n", *data_h);

    free(data_h);
    cudaFree(data_d);
    cudaFree(data_d);

    return 0;
}
