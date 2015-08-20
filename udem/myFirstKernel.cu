// includes, system
#include <stdio.h>
#include <assert.h>

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);

// Part 3 of 5: implement the kernel
__global__ void myFirstKernel(int *d_a)
{
	// We want each thread do the job, so we need
	// a global thread index to access elements of the array


	// Make each element of the array equal .e.g. 1000

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
    // pointer for host memory
    int *h_a;

    // pointer for device memory
    int *d_a;

    // define grid and block size
    int numBlocks = 8;
    int numThreadsPerBlock = 8;

    // Set the device to be used
    // Put the DeviceId assigned to you
    int DeviceId=0;
    cudaSetDevice(DeviceId);

    // Part 1 of 5: allocate host and device memory
    // Size of the array should be equal to the total number of threads
    size_t memSize = ;
    h_a = ;
    cudaMalloc();

    // Part 2 of 5: configure and launch kernel
    // Set up one dimensional grid and thread-block
    dim3 dimGrid();
    dim3 dimBlock();
    myFirstKernel<<<   >>>(d_a);

    // block until the device has completed
    cudaThreadSynchronize();

    // check if kernel execution generated an error
    checkCUDAError("kernel execution");

    // Part 4 of 5: device to host copy
    // Copy d_a to h_a
    cudaMemcpy();

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpy");

    // Part 5 of 5: verify the data returned to the host is correct
    // Use "assert" command to check that each element of the array equal to 1000 

    // free device memory
    cudaFree(d_a);

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
        exit(-1);
    }                         
}
