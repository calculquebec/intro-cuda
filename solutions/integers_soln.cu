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
    int *data_h;
    int *data_d;
    int i;
    const int N = 100;

 
    // allocate host and device memory
    data_h = ( int* ) malloc(N * sizeof(int));
    cudaMalloc(&data_d, N * sizeof(int));
 
    //Fill the array
    fillArray<<<N,1>>>( data_d, N );

    //Make sure the device has finished
    cudaThreadSynchronize();
    //Copy the results to the host
    cudaMemcpy(data_h, data_d, N*sizeof(int), cudaMemcpyDeviceToHost);
 
    // verify the data is correct
    for (i = 0; i < N; i++)
    {
        assert(data_h[i] == i );
    }
 
    // If the program makes it this far, then the results are
    // correct and there are no run-time errors.  Good work!
    printf("Correct!\n");
 
    free(data_h);
    cudaFree(data_d);
    return 0;
}

