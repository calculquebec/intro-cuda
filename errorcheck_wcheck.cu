//errorcheck_wcheck.cu: The program is designed to produce output
//'data = 7'. However, errors have been intentionally placed into 
//the program as an exercise in error checking.
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
  error = cudaMalloc((void**)&data_d, UINT_MAX*sizeof(int));
  if( error != cudaSuccess)
  {
    printf("cudaMalloc error: %s\n", cudaGetErrorString(error));
  }
  
  data_h = (int *)malloc(sizeof(int));

  setData<<<1,1>>>(0);
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
  cudaFree(data_d);
  return 0;
}
