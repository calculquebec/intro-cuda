#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define BLOCK_SIZE 4    // Number of threads in a block
#define MSIZE 8000         // Matrix dimension

/**
 *  Matrices are stored in row-major order:
 *    M(row, col) = M[row * MSIZE + col]
 */

__device__ unsigned int hash(unsigned int x)
{
    x = (x+0x7ed55d16) + (x<<12);
    x = (x^0xc761c23c) ^ (x>>19);
    x = (x+0x165667b1) + (x<<5);
    x = (x+0xd3a2646c) ^ (x<<9);
    x = (x+0xfd7046c5) + (x<<3);
    x = (x^0xb55a4f09) ^ (x>>16);
    return x;
}
__global__ void RandomFillKernel(float * A, unsigned int seed )
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    A[idx] = float(hash(idx+seed) / UINT_MAX);
}

/**
 * Check matrix product C = AB
 */
int checkProduct(float * A, float * B, float * C) {
    int i,j,k; //loop variables
    int fail = 0;
    float tol = 1e-2;
    float ABelement;

    //loop over rows 
    for (i = 0; i < MSIZE; i++) {
        //loop over columns
        for (j = 0; j < MSIZE; j++) {
            ABelement = 0.0f;
            //loop to compute matrix element
            for (k = 0; k < MSIZE; k++) {
                ABelement += A[i*MSIZE + k] * B[k*MSIZE + j];
            }
            // if matrix element is equal within tolerance
            if (fabsf(C[i*MSIZE + j] - ABelement) > tol) {
                printf("Matrix product problem: C != AB\n");
                printf("row %d col %d diff=%f\n", i,j,abs(C[i*MSIZE + j] - ABelement));
                fail = 1;
            }
            if (fail == 1) break;
        }
        if (fail == 1) break;
    }
    if (fail == 0) printf("Matrix product confirmed!\n");
    return fail;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(float* A, float* B, float* C) {
    // Each thread computes one element of C by accumulating results into Cvalue
    float Cvalue = 0;

    // Compute the thread index
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    // Compute the row and column
    for (int i = 0; i < MSIZE; ++i) {
        Cvalue += A[row * MSIZE + i] * B[i * MSIZE + col];
    }
    C[row*MSIZE+col] = Cvalue;
}
/**
 * Matrix multiplication.
 * Matrix dimensions are assumed to be multiples of BLOCK_SIZE
 */
void MatMul(float* A, float* B, float* C) {
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(MSIZE/BLOCK_SIZE,MSIZE/BLOCK_SIZE);
    MatMulKernel<<<dimGrid,dimBlock>>>(A, B, C);
    cudaDeviceSynchronize();
}

void print_duration(const char * msg, struct timespec * end)
{
    struct timespec start = *end;
    clock_gettime(CLOCK_MONOTONIC,end);
    struct timespec temp;
    if ((end->tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end->tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end->tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end->tv_sec-start.tv_sec;
        temp.tv_nsec = end->tv_nsec-start.tv_nsec;
    }
    double laps = temp.tv_sec + double(temp.tv_nsec)/1e9;
    printf(msg,laps);
}
int main(int argc, char** argv) {
    int N = 2;
    size_t size = MSIZE * MSIZE * sizeof(float);
    struct timespec ts;
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);

    float * matAs[N], * matAs_h[N];
    float * matBs[N], * matBs_h[N];
    float * matCs[N], * matCs_h[N];
    srand( time(NULL) );

    #pragma omp parallel for num_threads(5) private(ts)
    for (int d=0; d<N; d++)
    {
        cudaSetDevice(d % dev_count);

        clock_gettime(CLOCK_MONOTONIC,&ts);
        print_duration("Clock started. Duration %fs\n",&ts);
        
        // Allocate space for the matrices
        cudaMalloc(&matAs[d], size);
        cudaMalloc(&matBs[d], size);
        cudaMalloc(&matCs[d], size);
        matAs_h[d] = (float *)malloc(size);
        matBs_h[d] = (float *)malloc(size);
        matCs_h[d] = (float *)malloc(size);
        print_duration("Malloc done. Duration %fs\n",&ts);
        
        // Seed the random number generator
        RandomFillKernel<<<MSIZE*MSIZE/BLOCK_SIZE,BLOCK_SIZE>>>(matAs[d],rand());
        RandomFillKernel<<<MSIZE*MSIZE/BLOCK_SIZE,BLOCK_SIZE>>>(matBs[d],rand());
        print_duration("Initialization done. Duration %fs\n",&ts);
    
        //Multiply the matrices
        MatMul(matAs[d], matBs[d], matCs[d]);
        print_duration("Multiplication done. Duration %fs\n",&ts);
 
        //Copy the results back to host
        cudaMemcpy(matAs[d],matAs_h[d],size,cudaMemcpyDeviceToHost);
        cudaMemcpy(matBs[d],matBs_h[d],size,cudaMemcpyDeviceToHost);
        cudaMemcpy(matCs[d],matCs_h[d],size,cudaMemcpyDeviceToHost);
        print_duration("Copied data back. Duration %fs\n",&ts);

        //Check our work on the host
//        if (checkProduct(matAs_h[d], matBs_h[d], matCs_h[d]) != 0) {
//            printf("Your program may have errors\n");
//        }
//        print_duration("Checked data. Duration %fs\n",&ts);
    
        cudaFree(matCs[d]);
        cudaFree(matBs[d]);
        cudaFree(matAs[d]);
        free(matAs_h[d]);
        free(matBs_h[d]);
        free(matCs_h[d]);
        print_duration("Freed device memory. Duration %fs\n",&ts);
    }

    return 0;
}
