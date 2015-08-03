// includes, system
#include <stdio.h>
#include <assert.h>
#include <stdlib.h> 
 
__device__ void wasteTime(int n)
//This function wastes time proportional to n
{
  float temp;
  int i;
  for( i = 0; i < n; i++ )
    {
      temp = sin((float)i * 3.14f);
    }
}

__global__ void notDivergent(int n)
//This kernel should perform the same work as 
//divergent(), but the threads within a warp
//should not diverge
{
//  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float temp;
  //waste some time
  wasteTime(n);
  
}

__global__ void divergent(int n)
//This kernel should perform the same work as 
//notDivergent(), but the threads within
//a warp should be forced to diverge
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float temp;
  if ( tid % 2 == 0 )
    wasteTime(n);
  else
    wasteTime(n);
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

