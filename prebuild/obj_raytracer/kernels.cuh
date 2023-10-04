#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda.h"

void fillWithZeroesKernel(float* buf, int size);
void convolute_toeplitz_in_gpu(float* samples, float* IR, int ir_len, float* outputBuffer);
void convolute_fourier_in_gpu(float* samples, float* IR, float* outputBuffer);
void copy_from_gpu(float* device_pointer, float* host_pointer, size_t size);
void copy_to_gpu(float* host_pointer, float* device_pointer, size_t size);