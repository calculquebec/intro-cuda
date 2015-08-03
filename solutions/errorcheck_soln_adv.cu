//errorcheck_soln.cu: This program is designed to produce output
//'data = 7'. Error checking has been added and all errors have 
//been removed.
#include <stdio.h>
#include <stdlib.h>

#define CUDA_ERROR_EXIT_CODE 1

__global__ void setData(int *ptr)
{
  *ptr = 7;
}

static void checkCUDAError(cudaError_t error, const char * errTag)
{
  if ( error != cudaSuccess )
    {
      printf("Error - %s: %s\n", errTag, cudaGetErrorString( error ));
      exit( CUDA_ERROR_EXIT_CODE );
    }
}


int main(void)
{
  int *data_d = 0;
  int *data_h = 0;
//UINT_MAX is a huge number. The device runs out of memory.
  checkCUDAError( cudaMalloc((void**)&data_d, sizeof(int)), "cudaMalloc data_d" );
  data_h = (int *)malloc(sizeof(int));

//0 is a null pointer. The device tries to dereference a 
//null pointer producing an 'unspecified launch error'.
//This can be thought of as a CUDA segmentation fault.
  setData<<<1,1>>>(data_d);
  cudaThreadSynchronize();
  checkCUDAError( cudaGetLastError(), "setData kernel" );
  checkCUDAError( cudaMemcpy(data_h, data_d, sizeof(int), cudaMemcpyDeviceToHost), 
		  "cudaMemcpy error");
  printf("data = %d\n", *data_h);
  free(data_h);
  checkCUDAError( cudaFree(data_d), "cudaFree");
  return 0;
}
