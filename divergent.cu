// includes, system
#include <stdio.h>
#include <assert.h>
#include <stdlib.h> 
 

__global__ void notDivergent(int n)
//The threads should perform the same work as 
//in divergent(), but the threads within a warp
//should not diverge
{
}

__global__ void divergent(int n)
//The threads should perform the same work as 
//in notDivergent(), but the threads within
//a warp should be forced to diverge
{
}

// Program main
/////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
  const int N = 10000, threads = 10000;
  cudaEvent_t start, stop;
  float time;
  int nBlocks, nThreads;


    nThreads = 512;
    nBlocks = (threads + nThreads - 1)/nThreads;
 
    //Set up the timing variables and begin timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    //The Divergent Kernal
    divergent<<<nBlocks, nThreads>>>(N);

    //Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    //Compute the Elapsed Time
    cudaEventElapsedTime(&time, start, stop);


    printf("divergent kernel: %f milliseconds\n", time);
 
    //begin new timing
    cudaEventRecord(start, 0);
    
    //The non-Divergent Kernel
    notDivergent<<<nBlocks, nThreads>>>(N);

    //Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    //Compute the Kernel Time
    cudaEventElapsedTime(&time, start, stop);

    printf("non-divergent kernel: %f milliseconds\n", time);

    return 0;
}

