// includes, system
#include <stdio.h>
#include <assert.h>
#include <stdlib.h> 
 

__global__ void copy(float *data_in, float *data_out, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = 0;
  int xid = tid + offset;
  data_in[xid] = data_out[xid];
}
/////////////////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
  float *idata_h, *odata_h;
  float *idata_d, *odata_d;
  const int N = 1000;
  cudaEvent_t start, stop;
  float time, effBandwidth;
  int i;
 
    // allocate host and device memory
    idata_h = ( float* ) malloc(N * sizeof(float));
    odata_h = ( float* ) malloc(N * sizeof(float));
    cudaMalloc(&idata_d, N * sizeof(float));
    cudaMalloc(&odata_d, N * sizeof(float));
 
    //Fill the input array
    for (i = 0; i < N; i++)
      {
	idata_h[i] = (float) i;
      }

    //Copy the input array to the device
    cudaMemcpy(idata_d, idata_h, N*sizeof(float), cudaMemcpyHostToDevice);

    //Set up the timing variables and begin timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //Copy Kernal
    copy<<<N, 1>>>(idata_d, idata_h, N);

    //Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);


    //Copy the output array from the device
    cudaMemcpy(odata_h, odata_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // verify the data is correct
    for (i = 0; i < N; i++)
    {
        assert(odata_h[i] == idata_h[i] );
    }
 
    // If the program makes it this far, then the results are
    // correct and there are no run-time errors.  Good work!
    printf("Correct!\n");
 
    //Compute the Effective Bandwidth
    cudaEventElapsedTime(&time, start, stop);
    effBandwidth = 2*N*sizeof(float)/1.0e9/time;

    printf("Kernel time = %es\n", time);
    printf("Effective Bandwidth = %e s\n", effBandwidth);
     
    //Free the device and host memory
    free(idata_h); free(odata_h);
    cudaFree(idata_d); cudaFree(odata_d);
    return 0;
}

