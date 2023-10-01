#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda.h"

void fillWithZeroesKernel(float* buf, int size);
void read_from_gpu(float* device_pointer, float* host_pointer, size_t size);