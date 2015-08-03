#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#define SIZE 4000
float a[SIZE][SIZE];
float b[SIZE][SIZE];
float c[SIZE][SIZE];
float seq[SIZE][SIZE];
 

int main()
{
  int i,j,k;
  clock_t start, stop;   
  // Initialize matrices.
  for (i = 0; i < SIZE; ++i) {
    for (j = 0; j < SIZE; ++j) {
      a[i][j] = (float)i + j;
      b[i][j] = (float)i - j;
      c[i][j] = 0.0f;
    }
  }
  
  start = clock();
  // Compute matrix multiplication.
  #pragma acc kernels copyin(a,b) copy(c)
  for (i = 0; i < SIZE; ++i) {
    for (j = 0; j < SIZE; ++j) {
      for (k = 0; k < SIZE; ++k) {
	c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  stop = clock();
 
  printf("matrix multiplication completed! Matrix size = %d\n", SIZE);
  printf("Time required: %f s\n", (double) (stop - start) / CLOCKS_PER_SEC); 
  
  return 0;
}
