#include "./kernels.cuh"
#include <cufft.h>
#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void fillZeros(float *buf, int size)
{
    int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread < size)
        buf[thread] = 0.f;
}

void fillWithZeroesKernel(float *buf, int size)
{
    int numThreads = 256;
    int numBlocks;
    if (size % numThreads != 0) {
        numBlocks = (size / numThreads) + 1;
    } else {
        numBlocks = size / numThreads;
    }
    fillZeros<<<numBlocks, numThreads>>>(buf, size);
}

__global__ void vectorMultiply(float* a, float* b, float* c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        c[index] = a[index] * b[index];
    }
}

__global__ void convolute_toeplitz_lower_matrix(float* samples, float* IR, size_t ir_size, float* outputBuffer){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // where in the "matrix" are we
    int row, col;
    for (int i = 1; i <= ir_size; i++) {
        if (index < i) {
            row = i - 1;
            col = index;
            break;
        }
        index -= i;
    }
    
    /*
    because cell with col = row has I0
    then cell with col = row - 1 has I1 (because we moved column one to the left)
    then cell with col = row - x has Ix
    then row - col = x
    */
    int ir_index = row - col;
    atomicAdd(&outputBuffer[col], samples[row] * IR[ir_index]);
}

__global__ void convolute_toeplitz_lower_matrix_2d(float* samples, float* IR, size_t ir_size, float* outputBuffer) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= y && (x < ir_size && y < ir_size)) {
        atomicAdd(&outputBuffer[y*2], samples[x] * IR[y - x]);
        atomicAdd(&outputBuffer[(y*2) - 1], samples[x] * IR[y - x]);
    }
}

__global__ void convolute_toeplitz_vectors(float* samples, float* IR, size_t ir_size, float* outputBuffer){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int samples_offset = index / ir_size;
    int ir_index = index % ir_size;
    /*
    for this part of the convolution samples start at ir_len, thats why we sum ir_len to the samples index
    for each complete multiplication we (the whole IR vector * a subvector of samples) we need to continue with a new subvector
    this new subvector is the samples vector moved one unit forward, thats why we add a samples_offset to the samples index
    */
    atomicAdd(&outputBuffer[ir_size + samples_offset], samples[ir_size + samples_offset] * IR[ir_size - 1 - ir_index]);
}

void convolute_toeplitz_in_gpu(float* samples, float* IR, int ir_len, float* outputBuffer){
    //printf("ir_len: %d", ir_len);
    const int threadsPerBlock = 256;

    // first part, lower matrix multiplication
    /*
    cantidad operaciones = (ir_len * (ir_len + 1)) / 2
    esto es porque en la multiplicacion de la matriz triangular inferior
    la primera fila tiene una operacion, la segunda dos, y la n-esima (hasta el tope que es ir_len)
    equivale a Σ(i) de i=0 a i=n la cual se resuelve con ((n * (n + 1)) / 2)
    */
    int blocksPerGrid;
    if (((ir_len * (ir_len + 1)) / 2) < threadsPerBlock)
        blocksPerGrid = 1;
    else
        blocksPerGrid = (((ir_len * (ir_len + 1)) / 2) / threadsPerBlock) + 1;
    //convolute_toeplitz_lower_matrix<<<blocksPerGrid, threadsPerBlock>>>(samples, IR, ir_len, outputBuffer);
    dim3 threadsPorBlocks(32,32);
    int aaa = (ir_len / 32) + 1;
    dim3 numBlocks(aaa, aaa);
    convolute_toeplitz_lower_matrix_2d<<<numBlocks,threadsPorBlocks >>>(samples, IR, ir_len, outputBuffer);
    cudaDeviceSynchronize(); // necesario¿?

    // second part, vector multiplication
    size_t samples_size = sizeof(samples) / sizeof(float);
    /*
    ya procesamos ir_len celdas del output
    quedan (samples_size - ir_len) celdas restantes
    cada celda tiene ir_len multiplicaciones
    */
    //blocksPerGrid = ((samples_size - ir_len) * ir_len) / threadsPerBlock;
    //convolute_toeplitz_vectors<<<blocksPerGrid, threadsPerBlock>>>(samples, IR, ir_len, outputBuffer); // todo se precisa un offset
}

__global__ void convolute_fourier(float* samples, float* IR, float* outputBuffer){
    
}

void convolute_fourier_in_gpu(float* samples, float* IR, float* outputBuffer){ // WIP
    const int threadsPerBlock = 256;
    const int batchSize = 1; // Number of batches
    size_t samples_len = sizeof(samples) / sizeof(float);
    size_t ir_len = sizeof(samples) / sizeof(float);

    // Allocate device memory for samples
    cufftComplex* sampleData;
    cudaMalloc((void**)&sampleData, samples_len * sizeof(cufftComplex));
    
    for (int i = 0; i < samples_len; i++) {
        sampleData[i].x = samples[i];
        sampleData[i].y = 0.0f;
    }

    cufftHandle samplesPlan;
    cufftPlan1d(&samplesPlan, samples_len, CUFFT_C2C, batchSize);
    cufftExecC2C(samplesPlan, sampleData, sampleData, CUFFT_FORWARD);

    // Allocate device memory for IR
    cufftComplex* IRData;
    cudaMalloc((void**)&IRData, ir_len * sizeof(cufftComplex));
    
    for (int i = 0; i < ir_len; i++) {
        IRData[i].x = IR[i];
        IRData[i].y = 0.0f;
    }

    cufftHandle IRPlan;
    cufftPlan1d(&IRPlan, ir_len, CUFFT_C2C, batchSize);
    cufftExecC2C(IRPlan, IRData, IRData, CUFFT_FORWARD);

    // Convolute
    cufftComplex* resultData = new cufftComplex[N];
    // (a + ib) (c + id) = (ac – bd) + i(ad + bc)
    for (int i = 0; i < N; i++) {
        resultData[i].x = sampleData[i].x * IRData[i].x - sampleData[i].y * IRData[i].y;
        resultData[i].y = sampleData[i].x * IRData[i].y + sampleData[i].y * IRData[i].x;
    }

    // Invert result
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, batchSize);

    // Clean up
    cufftDestroy(samplesPlan);
    cufftDestroy(IRPlan);
    cudaFree(sampleData);
    cudaFree(IRData);
}

void copy_from_gpu(float* device_pointer, float* host_pointer, size_t size) {
    CUDA_CHK(cudaMemcpy(host_pointer, device_pointer, size, cudaMemcpyDeviceToHost));
};

void copy_to_gpu(float* host_pointer, float* device_pointer, size_t size) {
    CUDA_CHK(cudaMemcpy(device_pointer, host_pointer, size, cudaMemcpyHostToDevice));
};