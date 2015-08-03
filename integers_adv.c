// includes, system
#include <stdio.h>
#include <assert.h>
#include <stdlib.h> 
 

void fillArray(int *data, int N)
{
  int i;
  for( i = 0; i < N; i++)
    {
      data[i] = i;
    }
}
/////////////////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
    int *data;
    int i;
    const int N = 100;

 
    // allocate host memory
    data = ( int* ) malloc(N * sizeof(int));

    //Fill the array
    fillArray( data, N );
 
 
    // verify the data is correct
    for (i = 0; i < N; i++)
    {
        assert(data[i] == i );
    }
 
    // If the program makes it this far, then the results are
    // correct and there are no run-time errors.  Good work!
    printf("Correct!\n");
 
    free(data);
    return 0;
}

