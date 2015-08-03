#include <stdio.h>

const int N = 33 * 1024;
const int threadsPerBlock = 256; 
const int blocksPerGrid = ( (N+threadsPerBlock-1) / threadsPerBlock );

static void HandleError( cudaError_t err )
{
  if (err != cudaSuccess) {
    printf( "%s \n", cudaGetErrorString( err ));
    exit( 1 );
  }
}

__global__ void dot( float *a, float *b, float *c ) { 
   __shared__ float cache[threadsPerBlock];
   int idx = threadIdx.x + blockIdx.x * blockDim.x; 
   int cacheIndex = threadIdx.x;
  
  // set the cache values
  cache[cacheIndex] = a[idx] * b[idx];
  
  // synchronize threads in this block
  __syncthreads();
  // for reductions, threadsPerBlock must be a power of 2 
  // because of the following code
  int i = blockDim.x/2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2; 
  }
  if (cacheIndex == 0) 
     c[blockIdx.x] = cache[0];
}


int main( void ) {
  float *a, *b, c, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;

  // allocate memory on the CPU side
  a = (float*)malloc( N*sizeof(float) );
  b = (float*)malloc( N*sizeof(float) );
  partial_c = (float*)malloc( blocksPerGrid*sizeof(float) );

  // allocate the memory on the GPU
  HandleError( cudaMalloc( (void**)&dev_a, 
			    N*sizeof(float) ) );
  HandleError( cudaMalloc( (void**)&dev_b,
			    N*sizeof(float) ) );
  HandleError( cudaMalloc( (void**)&dev_partial_c, 
			    blocksPerGrid*sizeof(float) ) );

  // fill in the host memory with data
  for (int i=0; i<N; i++) 
  { 
    a[i] = i;
    b[i] = i*2; 
  }

  // copy the arrays ‘a’ and ‘b’ to the GPU
  HandleError( cudaMemcpy( dev_a, a, N*sizeof(float), 
			    cudaMemcpyHostToDevice ) );
  HandleError( cudaMemcpy( dev_b, b, N*sizeof(float), 
			    cudaMemcpyHostToDevice ) );
  dot<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b,
						   dev_partial_c );

 // copy the array 'c' back from the GPU to the CPU
 HandleError( cudaMemcpy( partial_c, dev_partial_c, 
			   blocksPerGrid*sizeof(float),
			   cudaMemcpyDeviceToHost ) );

 // finish up on the CPU side
 c = 0;
 for (int i=0; i<blocksPerGrid; i++) 
 {
   c += partial_c[i];
 }

 #define sum_squares(x) (x*(x+1)*(2*x+1)/6) 
 printf( "Does GPU value %.6g = %.6g ?\n", c,
	2 * sum_squares( (float)(N - 1) ) );


 // free memory on the GPU side
 HandleError( cudaFree( dev_a ) );
 HandleError( cudaFree( dev_b ) );
 HandleError( cudaFree( dev_partial_c ) );

 // free memory on the CPU side
 free( a );
 free( b );
 free( partial_c );
}
