#include <stdio.h>
#include <assert.h>
#include <stdlib.h> 

#define BLOCK_SIZE 128
**TODO: Transform into a kernel
**hint add __global__, compute index, remove the loop
void fillArray(int *data, int N) {
    int i;
    for( i = 0; i < N; i++) {
        data[i] = i;
    }
}

/**
 * Program main
 */
int main(int argc, char** argv) {
    int *data_h = 0;
    int *data_d = 0;
    const int N = 100;
 
    // Allocate host memory
    data_h = (int*)malloc(N * sizeof(int));
    cudaMalloc(&data_d,N * sizeof(int));

    // Fill the array
    **TODO: Replace by a kernel call
    fillArray(data, N);

    **TODO: Copy memory from device to host

    // verify the data is correct
    for (int i = 0; i < N; i++) {
        assert(data[i] == i);
    }
 
    // If the program makes it this far, then the results are
    // correct and there are no run-time errors.  Good work!
    printf("Correct!\n");
 
    free(data);
    cudaFree(data_d);
    return 0;
}
