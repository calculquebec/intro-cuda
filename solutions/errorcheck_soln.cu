//errorcheck_soln.cu: This program is designed to produce output
//'data = 7'. Error checking has been added and all errors have 
//been removed.
#include <stdio.h>
#include <stdlib.h>

__global__ void setData(int *ptr)
{
  *ptr = 7;
}


int main(void)
{
  int *data_d = 0;
  int *data_h = 0;
  cudaError_t error;
//UINT_MAX is a huge number. The device runs out of memory.
  error = cudaMalloc((void**)&data_d, sizeof(int));
  if( error != cudaSuccess)
  {
    printf("cudaMalloc error: %s\n", cudaGetErrorString(error));
  }
  
  data_h = (int *)malloc(sizeof(int));

//0 is a null pointer. The device tries to dereference a 
//null pointer producing an 'unspecified launch error'.
//This can be thought of as a CUDA segmentation fault.
  setData<<<1,1>>>(data_d);
  cudaThreadSynchronize();
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    printf("setData error: %s\n", cudaGetErrorString(error));
  }
  error = cudaMemcpy(data_h, data_d, sizeof(int), cudaMemcpyDeviceToHost);
  if(error != cudaSuccess)
  {
    printf("cudaMemcpy error: %s\n", cudaGetErrorString(error));
  }
  printf("data = %d\n", *data_h);
  free(data_h);
  //We only need to free data_d once. After this, it is no
  //longer a CUDA device pointer, and cant be cudaFree()'d again.
  cudaFree(data_d);
  return 0;
}
