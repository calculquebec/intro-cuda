#include <stdio.h>
#include <assert.h>

void checkCUDAError(const char *msg);


int main(){

int count,size;

float *da;
cudaDeviceProp *pDeviceProp;
	cudaGetDeviceCount(&count);
	printf("count=%d\n",count);

	size=20*sizeof(float);
	cudaMalloc((void**)&da,size);
	checkCUDAError("cudaMemcpy calls");
	pDeviceProp = (cudaDeviceProp*) malloc(sizeof(cudaDeviceProp));

	cudaSetDevice(6);
        checkCUDAError("Error setting a device\n");
	cudaGetDeviceProperties(pDeviceProp,6);

	printf( "Device Name \t – %s ", pDeviceProp->name );
	printf( "\n**************************************");
	printf( "\nTotal Global Memory\t\t -%d KB", pDeviceProp->totalGlobalMem/1024 );
	printf( "\nShared memory available per block \t – %d KB", pDeviceProp->sharedMemPerBlock/1024 );
	printf( "\nNumber of registers per thread block \t – %d", pDeviceProp->regsPerBlock );
	printf( "\nWarp size in threads \t – %d", pDeviceProp->warpSize );
	printf( "\nMemory Pitch \t – %d bytes", pDeviceProp->memPitch );
	printf( "\nMaximum threads per block \t – %d", pDeviceProp->maxThreadsPerBlock );
	printf( "\nMaximum Thread Dimension (block) \t – %d %d %d", pDeviceProp->maxThreadsDim[0], pDeviceProp->maxThreadsDim[1], pDeviceProp->maxThreadsDim[2] );
	printf( "\nMaximum Thread Dimension (grid) \t – %d %d %d", pDeviceProp->maxGridSize[0], pDeviceProp->maxGridSize[1], pDeviceProp->maxGridSize[2] );
	printf( "\nTotal constant memory \t – %d bytes", pDeviceProp->totalConstMem );
	printf( "\nCUDA ver \t – %d.%d", pDeviceProp->major, pDeviceProp->minor );
	printf( "\nClock rate \t – %d KHz", pDeviceProp->clockRate );
	printf( "\nTexture Alignment \t – %d bytes", pDeviceProp->textureAlignment );
//	printf( "\nDevice Overlap \t – %s", pDeviceProp-> deviceOverlap);
	printf( "\nNumber of Multi processors \t – %d", pDeviceProp->multiProcessorCount );


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
