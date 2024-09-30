#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "cuda.h"
#include "CircularBuffer.h"
#include <stdlib.h>
#include <windows.h>
#include <dbghelp.h>

#pragma comment(lib, "Dbghelp.lib")

void fillWithZeroesKernel(float *buf, int size);
void fillWithZeroesKernelDoubles(double* buf, int size);
void convolute_toeplitz_in_gpu(float *samples, float *IR, int ir_len, float *outputBuffer);
void convoluteFromLiveInput(double *samples, double *IR, unsigned int len, double *outputBuffer);
void convoluteFromAudioBuffer(float *samples, float *IR, unsigned int samples_len, unsigned int sample_rate, unsigned int ir_len, float *outputBuffer);
void copy_from_gpu(void *device_pointer, void *host_pointer, size_t size);
void copy_to_gpu(void *host_pointer, void *device_pointer, size_t size);
bool checkArrayZero(float *IR, unsigned int ir_len);
void zipArrays(double *d_outputBuffer_left, double *d_outputBuffer_right, double *d_outputBuffer, int monoBufferLength);
void normalizeBuffers(double *d_outputBuffer_left, double *d_outputBuffer_right, int monoBufferLength, int value);
void castFloatArrayToDouble(const float *input, double *output, size_t length);
void addIRsKernel(int ir_length, float *ir_left, float *ir_right);
