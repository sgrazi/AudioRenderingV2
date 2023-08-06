#include <cuda_runtime.h>
#include "./kernels.cuh"

__global__ void fillZeros(float *buf)
{
    *buf = 0.0f;
}

void fillWithZeroesKernel(float *buf)
{
    fillZeros<<<1, 1>>>(buf);
}