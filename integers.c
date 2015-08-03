// includes, system
#include <stdio.h>
#include <assert.h>
#include <stdlib.h> 
 
//WORKSHOP: Change this function to a CUDA kernel
void fillArray(int *data, int N)
{
  int i;
  for( i = 0; i < N; i++)
    {
      data[i] = i;
    }
}
/////////////////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
  //WORKSHOP: Declare data pointers for host and device arrays
  // (not necessary if using Unified memory)
    int *data;
    int i;
    const int N = 100;

 
    // allocate host memory
    data = ( int* ) malloc(N * sizeof(int));
    //WORKSHOP: Allocate device memory
    // Remove the host allocation above and use cudaMallocManaged() 
    // to allocate on host and device if using unified memory
 
    //Fill the array
    //WORKSHOP: Change this function call to a CUDA kernel call
    fillArray( data, N );

    //WORKSHOP: Make sure the device has finished
    //WORKSHOP: Copy the results to the host
    // (not necessary if using unified memory)

    // verify the data is correct
    for (i = 0; i < N; i++)
    {
        assert(data[i] == i );
    }
 
    // If the program makes it this far, then the results are
    // correct and there are no run-time errors.  Good work!
    printf("Correct!\n");
 
    free(data);
    //WORKSHOP: Free the device memory
    // (if using unified memory, you can free the host and device
    //  memory with one cudaFree() call)
    return 0;
}

