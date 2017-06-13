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
    // At best, our K20 card can handle 49152 bytes of shared storage (see deviceQuery output)
    // That means 49152/sizeof(float) elements, which is 12288.
    __shared__ float cache[BLOCK_SIZE];

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    cache[threadIdx.x] = (idx < N) ? a[idx]*b[idx] : 0.0f; // Put in 0.0 if we are out of bound

    __syncthreads(); // Wait for all threads to finish their multiplication

    for (int i = blockDim.x/2; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x+i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        c[blockIdx.x] = cache[0];
    }
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

    dotproduct<<<GRID_SIZE, BLOCK_SIZE>>>(dev_a, dev_b, dev_c, SIZE);
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
    for (int i=0; i<GRID_SIZE; i++)
        result += h_c[i];

    printf("Result: %f\n", result);

cleanup:
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(h_c);

    return ret;
}
