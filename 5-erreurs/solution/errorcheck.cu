#include <stdio.h>
#include <stdlib.h>

__global__ void setData(int *ptr) {
    *ptr = 7;
}


int main(int, char**) {
    int *data_d = 0;
    int *data_h = 0;

    cudaError_t err;

    if ((err = cudaMalloc((void**)&data_d, sizeof(int))) != cudaSuccess) {
        printf("Could not allocate that much memory. \n%s",cudaGetErrorString(err));
        exit(1);
    }
    data_h = (int *)malloc(sizeof(int));

    setData<<<1,1>>>(0);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error calling setData. \n%s",cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(data_h, data_d, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Could not copy memory \n%s",cudaGetErrorString(err));
        goto cleanup;
    }

    printf("data = %d\n", *data_h);
    free(data_h);

cleanup:
    if ((err = cudaFree(data_d)) != cudaSuccess) {
        printf("Could not free memory (free #1) \n%s",cudaGetErrorString(err));
        exit(1);
    }

    if ((err = cudaFree(data_d)) != cudaSuccess) {
        printf("Could not free memory (free #2) \n%s",cudaGetErrorString(err));
        exit(1);
    }

    return 0;
}
