// In this example a HOST array a[] is copied to Device array d_a, then Kernel is launched
// The Kernel fill up the d_a[] array with some numbers. Then result is copied back to Host array a[]
// Make these 3 operations concurrent, i.e. run them in 4 different streams.


// WHAT is DIFFERENT from example Stream1 ?  : Modify the STEP 6 in a such a way that operations
// are performed in the following order:  H2D copies for all streams, kernels for all streams, D2H copies for all streams

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

# define SIZE 4096

// STEP 4 of 8: Implement a KERNEL
// Write a CUDA kernel that fill up an array with some integers numbers,
// make the kernel more time consuming by including a loop that just burns time 
__global__ void kernel(int *d_a,int offset){
	// Make all the threads and blocks work. Create a "global" thread index 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Put data in the array
	for(int i=0;i<1000;i++) d_a[offset+idx]=idx*2;
}

void checkCUDAError(const char *msg);
int main(int argc, char **argv)
{
    	int ndevices;
	int cuda_device = 1;
    	int nstreams = 4;
	int streamSize;	

	int *a;
	int *d_a;

	// Count number of GPUs on board
	cudaGetDeviceCount(&ndevices);
	checkCUDAError("cudaGetDevice failed !");
	printf("Number of GPUs available to run = %d\n",ndevices);

	// Set the GPU device
        cudaSetDevice(0);
	checkCUDAError("cudaSetDevice failed !");


	// Check CUDA Device properties for whether overlap between kernel & memcpy is supported
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, cuda_device);
	printf("Device: <%s> canMapHostMemory: %s\n", deviceProp.name, deviceProp.canMapHostMemory ? "Yes" : "No");
	printf("Number of copy engines = %d\n",deviceProp.asyncEngineCount);

	// STEP 1 of 8: Allocate memory of GPU
	// CUDA memory allocation
	cudaMalloc((void**)&d_a,SIZE*sizeof(int));
	checkCUDAError("cudaMalloc failed !");

	// STEP 2 of 8: Allocate pinned memory on HOST
	// Paged-locked memory allocation of HOST
	cudaHostAlloc((void**)&a,SIZE*sizeof(int),cudaHostAllocDefault);
	checkCUDAError("cudaHostAlloc failed !");

	//STEP 3 of 8: Create CUDA streams
    	cudaStream_t *streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
    	for(int i=0;i<nstreams;i++) {
		cudaStreamCreate(&streams[i]);
		checkCUDAError("cudaStreamCreate failed !");
	}

	//generate data
	for(int i=0;i<SIZE;i++) a[i]=0;

	// STEP 5 of 8: Create CUDA grid (blocks, threads) in a such a way that the arrays a[] or d_a[]
	// are handled in chunks. Number of chunks = number of streams, so each stream makes the copies 
	// and perform kernel operations only on its own chunk.
	streamSize = SIZE/nstreams;
	int numthreads=512;
        int numblocks=streamSize/numthreads;
	dim3 dimGrid(numblocks);
	dim3 dimBlock(numthreads);

	// STEP 6 of 8: Main LOOP that should include D2H, H2D copies, and a Kernel invocation
	for(int i=0;i<nstreams;i++){
		int offset = i*streamSize;
		cudaMemcpyAsync(&d_a[offset],&a[offset],streamSize,cudaMemcpyDeviceToHost,streams[i]);
	}

	for(int i=0;i<nstreams;i++){
                int offset = i*streamSize;
		kernel<<<dimGrid, dimBlock, 0, streams[i]>>>(d_a,offset);
	}
	
	for(int i=0;i<nstreams;i++){
                int offset = i*streamSize;
		cudaMemcpyAsync(&a[offset],&d_a[offset],streamSize,cudaMemcpyHostToDevice,streams[i]);
	}

	// STEP 7 of 8: Synchronize streams
	for(int i=0;i<nstreams;i++) {
		cudaStreamSynchronize(streams[i]);
	}



	// STEP 8 of 8: Destroy streams
	for(int i=0;i<nstreams;i++) {
                cudaStreamDestroy(streams[i]);
	}

}


void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }
}
