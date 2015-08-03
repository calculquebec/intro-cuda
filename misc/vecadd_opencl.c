// Code copied by hand from Heterogeneous Computing with OpenCL
// by Gaster, et. al. Page 32-38

//compile with:
//module add CUDA_Toolkit
//nvcc -lOpenCL -o vecadd vecadd_opencl.c

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <CL/cl.h>


// Character Array containing the kernel source code
const char* programSource =
  "__kernel                                                          \n"
  "void vecadd(__global int *A,                                    \n"
  "            __global int *B,                                      \n"
  "            __global int *C)                                      \n"
  "{                                                                 \n"
  "                                                                  \n"
  "  int idx = get_global_id(0);                                     \n"
  "  C[idx] = A[idx] + B[idx];                                       \n"
  "}                                                                 \n";

int main()
{
  int *A = NULL;  //Input Array
  int *B = NULL;  //Input Array
  int *C = NULL;  //Output Array
  int i;

  const int elements = 2048;

  size_t datasize = sizeof(int)*elements;

  A = (int*)malloc(datasize);  //Input Array
  B = (int*)malloc(datasize);  //Input Array
  C = (int*)malloc(datasize);  //Output Array

  //Initialize the input data
  for(i = 0; i < elements; i++)
    {
      A[i] = i;
      B[i] = i;
    }
  cl_int status;

  //Step 1: Discover and initialize the platforms
  cl_uint numPlatforms = 0;
  cl_platform_id *platforms = NULL;

  status = clGetPlatformIDs(0, NULL, &numPlatforms);

  platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));

  status = clGetPlatformIDs(numPlatforms, platforms, NULL);


  //Step 2: Discover and Initialize the devices
  cl_uint numDevices = 0;

  cl_device_id *devices = NULL;

  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, 
			  &numDevices);
  devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));

  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, 
			  devices, NULL);
  
  //Step 3: Create a Context
 
  cl_context context = NULL;

  context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

  //Step 4: Create a Command Queue

  cl_command_queue cmdQueue;
  cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
  
  //Step 5: Create device buffers

  cl_mem bufferA;
  cl_mem bufferB;
  cl_mem bufferC;

  bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
  bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
  bufferC = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);


  //Step 6: Write host data to device buffers

  status = clEnqueueWriteBuffer(cmdQueue, bufferA, CL_FALSE, 0, datasize,
			        A, 0, NULL, NULL);
  status = clEnqueueWriteBuffer(cmdQueue, bufferB, CL_FALSE, 0, datasize, 
				B, 0, NULL, NULL);

  //Step 7: Create and compile the program

  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource,
						 NULL, &status);

  status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

  //Step 8: Create the kernel

  cl_kernel kernel = NULL;

  kernel = clCreateKernel(program, "vecadd", &status);

  //Step 9: Set the kernel arguments

  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
  status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

  //Step 10: Configure the work-item structure

  size_t globalWorkSize[1];

  globalWorkSize[0] = elements;

  //Step 11: Enqueue the kernel for execution

  status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize,
				  NULL, 0, NULL, NULL);

  //Step 12: Read the output buffer back to the host 

  clEnqueueReadBuffer(cmdQueue, bufferC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);

  bool result = true;

  for(i = 0; i < elements; i++)
    {
      if (C[i] != i+i)
	{
	  result = false;
	  break;
	}
    }
  if(result)
    {
      printf("Output is correct \n");
    }
  else
    {
      printf("Output is incorrect\n");
    }

  //Step 13: Release OpenCL Resources

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(cmdQueue);
  clReleaseMemObject(bufferA);
  clReleaseMemObject(bufferB);
  clReleaseMemObject(bufferC);
  clReleaseContext(context);

  free(A);
  free(B);
  free(C);
  free(platforms);
  free(devices);
}
