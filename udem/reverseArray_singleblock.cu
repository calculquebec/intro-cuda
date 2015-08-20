// includes, system
#include <stdio.h>
#include <assert.h>

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

// Part 5: implement the kernel
__global__ void reverseArrayBlock(int *d_out, int *d_in)
{
    // create array original and reverse indices
    int in = ;
    int out = ;

    // reverse the array content using appropriate indices
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
    // pointer for host memory and size
    int *h_a;
    int dimA = 256;

    // pointer for device memory
    int *d_b, *d_a;

    // define grid and block size
    int numBlocks = 1;
    int numThreadsPerBlock = dimA;

    // Set the device to be used
    // Put the DeviceId assigned to you
    int DeviceId=0;
    cudaSetDevice(DeviceId);

    // Part 1: allocate host and device memory
    size_t memSize = numBlocks * numThreadsPerBlock * sizeof(int);
    h_a = ;
    cudaMalloc( );
    cudaMalloc( );

    // Part 2: Initialize input array on host

    // Part 3: Copy host array to device array
    cudaMemcpy( );

    // Part 4: launch kernel
    dim3 dimGrid();
    dim3 dimBlock();
    reverseArrayBlock<<<  >>>( d_b, d_a );

    // block until the device has completed
    cudaThreadSynchronize();

    // check if kernel execution generated an error
    // Check for any CUDA errors
    checkCUDAError("kernel invocation");

    // Part 6: device to host copy
    cudaMemcpy( );

    // Check for any CUDA errors
    checkCUDAError("memcpy");

    // Part 7: verify the data returned to the host is correct

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
