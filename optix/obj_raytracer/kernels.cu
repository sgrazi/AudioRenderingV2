#include <cuda_runtime.h>
#include "./kernels.cuh"

__global__ void fillZeros(float *buf)
{
    int thread = blockDim.x * blockIdx.x + threadIdx.x;
    buf[thread] = 0.0f;
}

void fillWithZeroesKernel(float *buf, int size)
{
    int numThreads = 1024;
    int numBlocks;
    if (size % numThreads != 0) {
        numBlocks = (size / numThreads) + 1;
    } else {
        numBlocks = size / numThreads;
    }
    fillZeros<<<numThreads, numBlocks>>>(buf);
}