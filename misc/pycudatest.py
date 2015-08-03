import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float64))
a_doubled = (2*a_gpu).get()
print a_doubled - a_gpu
print a_gpu
