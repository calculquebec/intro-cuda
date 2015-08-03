#include <stdio.h>
#include <math.h>

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

int checkProduct(const Matrix A, const Matrix B, const Matrix C)
//Check matrix product C = AB
{
  int i,j,k; //loop variables
  int fail = 0;
  float tol = 1e-2;
  float ABelement;

  if ( A.width != B.height || A.height != C.height || B.width != C.width)
    {
      printf("checkProduct failed basic tests\n");
      fail = 1;
    }
  else
    {
      for (i = 0; i < C.height; i++)
	{
	  for (j = 0; j < C.width; j++)
	    {
	      ABelement = 0.0f;
	      for (k = 0; k < A.width; k++)
		{
		  ABelement += *(A.elements + i*A.width + k)
		    * *(B.elements + k*B.width + j);
		}
	      if (fabsf(*(C.elements + i*C.width + j) - ABelement) > tol)
		{
		  printf("Matrix product problem: C != AB\n");
		  printf("row %d col %d diff=%f\n", i,j,abs(*(C.elements + i*C.width + j) - ABelement));
		  fail = 1;
		}
	      if (fail == 1) break;
	    }
	  if (fail == 1) break;
	}
    }
  if (fail == 0) printf("Matrix product confirmed!\n");
  return fail;
} 
