#include <stdio.h>
#include <assert.h>
#include <stdlib.h> 
 
#define BLOCK_SIZE 128
__global__ void fillArray(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] = idx;
    }
}
/**
 * Program main
 */
int main(int argc, char** argv) {
    int *data_h = 0;
    int *data_d = 0;
    const size_t N = 100;
 
    // Allocate host and device memory
    data_h = (int*)malloc(N * sizeof(int));
    cudaMalloc(&data_d,N * sizeof(int));
 
    // Fill the array
    fillArray<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(data_d, N);

    // Make sure the device has finished
    cudaDeviceSynchronize();
    // Copy the results to the host
    cudaMemcpy(data_h, data_d, N*sizeof(int), cudaMemcpyDeviceToHost);
 
    // Verify the data is correct
    for (int i = 0; i < N; ++i) {
        assert(data_h[i] == i);
    }
 
    // If the program makes it this far, then the results are
    // correct and there are no run-time errors.  Good work!
    printf("Correct!\n");
 
    free(data_h);
    cudaFree(data_d);
    return 0;
}
