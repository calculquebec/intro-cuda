// includes, system
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

int main( int argc, char** argv) 
{
    int DeviceId=1;
    int numdevices;

    // STEP 1: Get number of devices available
    cudaGetDeviceCount(&numdevices);

    // STEP 2: Set the Device you will work with
    cudaSetDevice(DeviceId);

    // STEP 3: Get the last error message printed out
    printf("Checking last error: %s\n",cudaGetErrorString(cudaGetLastError()));

}

