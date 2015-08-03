// includes, system
#include <stdio.h>
#include <assert.h>
#include <stdlib.h> 
 

__global__ void fillArray(int *data, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    data[idx] = idx;
}
/////////////////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
    int *data;
    int i;
    const int N = 100;

 
    // allocate unified memory
    cudaMallocManaged(&data, N);
 
    //Fill the array
    fillArray<<<N,1>>>( data, N );
    //Synchronize data between host and device
    cudaDeviceSynchronize();

    // verify the data is correct
    for (i = 0; i < N; i++)
    {
        assert(data[i] == i );
    }
 
    // If the program makes it this far, then the results are
    // correct and there are no run-time errors.  Good work!
    printf("Correct!\n");
 
    //Free the Cuda Managed memory
    cudaFree(data);
    return 0;
}

