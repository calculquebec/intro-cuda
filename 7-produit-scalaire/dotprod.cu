#include <cuda.h>
#include <stdio.h>
#include <device_functions.h>

const int SIZE = 1000000;      // Vector sizes
const int BLOCK_SIZE = 1024;    // Threads per block
const int GRID_SIZE = ((SIZE+BLOCK_SIZE-1) / BLOCK_SIZE);   // Number of blocks

/**
 * Fill vectors directly on the GPU.
 */
__global__ void fillvectors(float *a, float *b, int N) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < N) {
        a[idx] = idx;
        b[idx] = idx*2;
    }
}

/**
 * Compute the dot product of two vectors a and b.
 */
__global__ void dotproduct(float *a, float *b, float *c, int N) {
    // Use __shared__ memory to cache result within a block
    // How much memory can we allocate in shared storage ? 
    // What is the maximum block size ?

    // compute the index
    // compute the multiplication in shared memory



    // perform the reduction
    // Can it be done completely on the GPU in one kernel call ?
}

int main(int argc, char **argv) {
    int ret = 0;

    float result = 0.0f;
    float *dev_a = 0, *dev_b = 0, *dev_c = 0, *h_c = 0;

    size_t size = SIZE*sizeof(float);

    h_c = (float *)malloc(GRID_SIZE*sizeof(float));
    if (cudaMalloc(&dev_a, size) != cudaSuccess ||
        cudaMalloc(&dev_b, size) != cudaSuccess ||
        cudaMalloc(&dev_c, GRID_SIZE*sizeof(float)) != cudaSuccess)
    {
        ret = 1;
        printf("Error allocating GPU memory.\n");
        goto cleanup;
    }

    fillvectors<<<GRID_SIZE, BLOCK_SIZE>>>(dev_a, dev_b, SIZE);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        ret = 3;
        printf("Error filling vectors.\n");
        goto cleanup;
    }

    // Call to the kernel


    if (cudaDeviceSynchronize() != cudaSuccess) {
        ret = 5;
        printf("Error computing a dotproduct.\n");
        goto cleanup;
    }

    if (cudaMemcpy(h_c, dev_c, GRID_SIZE*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        ret = 6;
        printf("Could not copy result back to host.\n");
        goto cleanup;
    }
    // Complete the reduction

    printf("Result: %f\n", result);

cleanup:
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(h_c);

    return ret;
}
