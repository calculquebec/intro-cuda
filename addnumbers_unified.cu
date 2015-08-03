#include<stdio.h>

__global__ void add2(int *a)
{
    *a = *a + 2;
}

int main( void )
{
    int *data;
    cudaMallocManaged(&data, sizeof(int));
    
    *data = 5;
    add2<<<1,1>>>(data);
    cudaDeviceSynchronize();
    printf("data: %d\n", *data);
    cudaFree(data);
    return 0;    
}