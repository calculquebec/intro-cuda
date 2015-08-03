#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define BLOCK_SIZE 4    // Number of threads in a block
#define MSIZE 4         // Matrix dimension

/**
 *  Matrices are stored in row-major order:
 *    M(row, col) = M[row * MSIZE + col]
 */

__global__ void MatMulKernel(float *, float *, float *);

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

/**
 * Matrix multiplication.
 * Matrix dimensions are assumed to be multiples of BLOCK_SIZE
 */
void MatMul(float* A, float* B, float* C) {
    float *d_A = 0;
    size_t size = MSIZE * MSIZE * sizeof(float);
    // Allocate space for matrix A on device
    cudaMalloc(&d_A, size);
    // Copy matrix A to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    float *d_B = 0;
    // Allocate space for matrix B on device
    cudaMalloc(&d_B, size);
    // Copy matrix B to device
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    float *d_C = 0;
    **TODO: Allocate space for C in device memory

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(MSIZE/BLOCK_SIZE,MSIZE/BLOCK_SIZE);
    MatMulKernel<<<dimGrid,dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    **TODO: Copy C matrix from device to host

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
     **TODO: Free device memory for C
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(float* A, float* B, float* C) {
    // Each thread computes one element of C by accumulating results into Cvalue
    float Cvalue = 0;

    // Compute the thread index
    **TODO: Compute the thread indexes: int col = ???  int row = ???

    // Compute the row and column
    for (int i = 0; i < MSIZE; ++i) {
        Cvalue += A[row * MSIZE + i] * B[i * MSIZE + col];
    }
    C[row*MSIZE+col] = Cvalue;
}

int main(int argc, char** argv) {
    float *matA = 0, *matB = 0, *matC = 0;
    int i, j; //row and column indices
    size_t size = MSIZE * MSIZE * sizeof(float);

    // Allocate space for the matrices
    matA = (float *) malloc(size);
    matB = (float *) malloc(size);
    matC = (float *) malloc(size);

    // Seed the random number generator
    srand( time(NULL) );

    // Generate a random value for each element of A and B
    for( i = 0; i < MSIZE; i++) {
        for( j = 0; j < MSIZE; j++) {
            matA[i * MSIZE + j] = rand() / (float) RAND_MAX;
            matB[i * MSIZE + j] = rand() / (float) RAND_MAX;
        }
    }

    //Multiply the matrices
    MatMul(matA, matB, matC);

    //Check our work on the host
    if (checkProduct(matA, matB, matC) != 0) {
        printf("Your program may have errors\n");
    }

    free(matC);
    free(matB);
    free(matA);

    return 0;
}
