// includes, system
#include <stdio.h>
#include <assert.h>

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

// Part 6: implement the kernel
__global__ void reverseArrayBlock( int* d_out, int* d_in )
{

    // create original and reverse array indices
    // keeping in mind that you have multiple blocks reversing the array content
    int in = ;
    int out = ;

    // reverse array using appropriate indices
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
    // pointer for host memory and size
    int *h_a;
    int dimA = 256 * 1024; // 256K elements (1MB total)

    // pointer for device memory
    int *d_b, *d_a;

    // define grid and block size
    int numThreadsPerBlock = 256;

    // Part 1: compute number of blocks needed based on array size and desired block size
    int numBlocks = ;  

    // Part 2: allocate host and device memory
    size_t memSize = numBlocks * numThreadsPerBlock * sizeof(int);
    h_a = (int *) malloc(memSize);
    cudaMalloc( );
    cudaMalloc( );

    // Part 3: Initialize input array on host

    // Part 4: Copy host array to device array
    cudaMemcpy( );

    // Part 5: Set up grid and launch kernel
    dim3 dimGrid( );
    dim3 dimBlock( );
    reverseArrayBlock<<<  >>>( d_b, d_a );

    // block until the device has completed
    cudaThreadSynchronize();

    // check if kernel execution generated an error
    // Check for any CUDA errors
    checkCUDAError("kernel invocation");

    // Part 7: device to host copy
    cudaMemcpy( );

    // Check for any CUDA errors
    checkCUDAError("memcpy");

    // Part 8: verify the data returned to the host is correct

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);

    // free host memory
    free(h_a);

    // If the program makes it this far, then the results are correct and
    // there are no run-time errors.  Good work!
    printf("Correct!\n");

    return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}
