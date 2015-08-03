#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// Thread block size
#define BLOCK_SIZE 4
#define MSIZE 4

// Matrices are stored in row-major order:
// M(row, col) = M[row * MSIZE + col]

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(float *, float *, float *);

int checkProduct(float * A, float * B, float * C)
//Check matrix product C = AB
{
  int i,j,k; //loop variables
  int fail = 0;
  float tol = 1e-2;
  float ABelement;

  //loop over rows 
  for (i = 0; i < MSIZE; i++)
     {
       //loop over columns
       for (j = 0; j < MSIZE; j++)
	  {
	    ABelement = 0.0f;
            //loop to compute matrix element
	    for (k = 0; k < MSIZE; k++)
	      {
		ABelement += A[i*MSIZE + k] * B[k*MSIZE + j];
	      }
	    //if matrix element is equal within tolerance
	      if (fabsf(C[i*MSIZE + j] - ABelement) > tol)
		{
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


// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(float* A, float* B, float* C)
{
    
    float *d_A;
    size_t size = MSIZE * MSIZE * sizeof(float);
    //allocate space for matrix A on device
    cudaMalloc(&d_A, size);
    //copy matrix A to device
    cudaMemcpy(d_A, A, size,
               cudaMemcpyHostToDevice);
    float *d_B;
    //allocate space for matrix B on device
    cudaMalloc(&d_B, size);
    //copy matrix B to device
    cudaMemcpy(d_B, B, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    float *d_C;
    cudaMalloc(&d_C, size);

    // Invoke kernel
    MatMulKernel<<<MSIZE * MSIZE / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C, d_C, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(float* A, float* B, float* C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    //compute the thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //compute the row and column
    int row = idx / MSIZE;
    int col = idx - row * MSIZE;
    for (int i = 0; i < MSIZE; ++i)
        Cvalue += A[row * MSIZE + i]
                * B[i * MSIZE + col];
    C[idx] = Cvalue;
}

int main(int argc, char** argv)
{
  float *matA, *matB, *matC;
  int i, j; //row and column indices
  uint size = MSIZE * MSIZE * sizeof(float);
  // Allocate space for the matrices
  matA = (float *) malloc(size);
  matB = (float *) malloc(size);
  matC = (float *) malloc(size);
  // Seed the random number generator
  srand( time(NULL) );

  // Generate a random value for each element of A and B
  for( i = 0; i < MSIZE; i++)
  {
    for( j = 0; j < MSIZE; j++)
    {
      matA[i * MSIZE + j] = rand() / (float) RAND_MAX;
      matB[i * MSIZE + j] = rand() / (float) RAND_MAX;
    }
  }

  //Multiply the matrices
  MatMul(matA, matB, matC);

  //Check our work on the host
  if (checkProduct(matA, matB, matC) != 0)
     printf("Your program may have errors\n");
  free(matA); free(matB); free(matC);
  return 0;

}
