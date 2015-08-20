// includes, system
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

#define WIDTH 8192
#define TILE_WIDTH 8

void checkCUDAError(const char *msg);
void fillRandomSingle(int m, int n, float* a, float min, float max);
double getHighResolutionTime(void);

__global__ void MatrixMultKernel(float *Md, float *Nd, float *Pd, int Width)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];	
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
  
	int m,k;
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	// Identify the row and column of the Pd element to work on
	int Row = by*TILE_WIDTH + ty;
	int Col = bx*TILE_WIDTH + tx;

	// Loop over the Md and Nd tiles for the Pd element
	float Pvalue = 0;
	for(m=0; m < Width/TILE_WIDTH; m++){
		//  Collaborative loading of Md and Nd tiles into shared memory
		Mds[tx][ty] = Md[(m*TILE_WIDTH + tx)*Width + Row];
		Nds[tx][ty] = Nd[Col*Width + (m*TILE_WIDTH + ty)];
		__syncthreads();
		for(k=0; k < TILE_WIDTH; k++)
			Pvalue += Mds[tx][k] * Nds[k][ty];
		__syncthreads ();
	}
	Pd [Row*Width + Col] = Pvalue;


}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	int DeviceId=0;
	cudaSetDevice(DeviceId);

	int Width, i,j,k;
	float *M;
	float *N;
	float *P;
	float *Md;
	float *Nd;
	float *Pd;

	Width=WIDTH;
	int size=Width*Width*sizeof(float);

	M = (float*)malloc(size);
	N = (float*)malloc(size);

	cudaMalloc((void**)&Md, size);
	cudaMalloc((void**)&Nd, size);

	fillRandomSingle(Width, Width, M, -10.0, 10.0);
	fillRandomSingle(Width, Width, N, -10.0, 10.0);
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
	checkCUDAError("Failed to copy data to GPU");

	// Allocate P on the host and device
	P = (float*)malloc(size);
	cudaMalloc((void**)&Pd, size);

	printf("Width=%d\n",Width);
	// Setup the execution kernel grid
	dim3 dimGrid(Width/TILE_WIDTH, Width/TILE_WIDTH);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	// Launch kernel
	double start_time = getHighResolutionTime();
	MatrixMultKernel<<< dimGrid, dimBlock, 2*sizeof(float)*TILE_WIDTH*TILE_WIDTH >>> (Md, Nd, Pd, Width);
	cudaThreadSynchronize();
	double end_time = getHighResolutionTime();
	printf("Exec.time=%f\n",end_time-start_time);
	checkCUDAError("Kernel failed");

	// Read Pd from the device 
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);

	// Free device matrices
	cudaFree(Md); cudaFree(Nd); cudaFree(Pd);

	return 0;
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

void fillRandomSingle(int m, int n, float* a, float min, float max)
{
    int i, j;

    srand(1);

    for (j=0; j<m; j++)
    {
        for (i=0; i<n; i++)
        {
            a[j*n+i] = min + (max-min) * rand()/RAND_MAX;
        }
    }
}

double getHighResolutionTime(void)
{
    struct timeval tod;

    gettimeofday(&tod, NULL);
    double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
    return time_seconds;
}
