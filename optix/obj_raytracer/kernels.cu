#include <cuda_runtime.h>
#include "kernels.cuh"

namespace kernels
{
    __global__ void fillZeros(float *buf)
    {
        *buf = 0.0f;
    }

    void fillWithZeroesKernel(float *buf)
    {
        #ifndef __INTELLISENSE__
        #def __INTELLISENSE__
        kernels::fillZeros< < <1, 1> > >(buf);
        #endif
    }
}