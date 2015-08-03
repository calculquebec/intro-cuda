//nvcc -o first -lcudart -lcuda -lcublas first.cu

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include<cublas_v2.h>
#include<cuda_runtime.h>

#define N (275)
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

int main(int argc, char** argv)
{
	cublasStatus_t status;
	float* h_A;
	float* h_B;
	float* h_C;
	float* d_A = 0;
	float* d_B = 0;
	float* d_C = 0;
	float alpha = 1.0f;
	float beta = 0.0f;
	int n2 = N*N;
	int i;
	cublasHandle_t handle;

	/* Initialize CUBLAS */

	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	}

	/* Allocate host memory for matrices */
	h_A = (float *)malloc(n2* sizeof(h_A[0]));
	h_B = (float *)malloc(n2 * sizeof(h_B[0]));
	h_C = (float *)malloc(n2 * sizeof(h_C[0]));

	/* Fill the matrices with test data */
	for (i=0; i<n2; i++)
	{
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
		h_C[i] = rand() / (float)RAND_MAX;
	}

	/* Allocate device memory for the matrices */
	if (cudaMalloc((void**)&d_A, n2 * sizeof(d_A[0])) != cudaSuccess)
	{
		fprintf (stderr, "!!!! device memory allocation error (allocate A)\n");
		return EXIT_FAILURE;
	}
	if (cudaMalloc((void**)&d_B, n2 * sizeof(d_B[0])) != cudaSuccess)
	{
		fprintf (stderr, "!!!! device memory allocation error (allocate B)\n");
		return EXIT_FAILURE;
	}
	if (cudaMalloc((void**)&d_C, n2 * sizeof(d_C[0])) != cudaSuccess)
	{
		fprintf (stderr, "!!!! device memory allocation error (allocate C)\n");
		return EXIT_FAILURE;
	}

	/* Initialize the device matrices with the host matrices */
	status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	}
	status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	}
	status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	}

	/* Performs operation using cublas */
//Single precision general matrix multiplication
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 
	       N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	}

	/* Allocate host memory for reading back the result from device memory */
	h_C = (float *)malloc(n2 * sizeof(h_C[0]));

	/* Read the result back */

	status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);

	/* Memory clean up */
	free(h_A);
	free(h_B);
	free(h_C);
	if(cudaFree(d_A) != cudaSuccess)
	{
		fprintf (stderr, "!!!! memory free error (A)\n");
		return EXIT_FAILURE;
	}
	if(cudaFree(d_B) != cudaSuccess)
	{
		fprintf (stderr, "!!!! memory free error (B)\n");
		return EXIT_FAILURE;
	}
	if(cudaFree(d_C) != cudaSuccess)
	{
		fprintf (stderr, "!!!! memory free error (C)\n");
		return EXIT_FAILURE;
	}

	/* Shutdown */
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("%s\n", cudaGetErrorString( cudaGetLastError() ) );
	}

	return EXIT_SUCCESS;



	

}
