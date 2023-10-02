#include "./kernels.cuh"
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
    int ir_index = row - col;
    */
    atomic_add(samples[col] * IR[ir_index], outputBuffer[col]);
}

__global__ void convolute_toeplitz_vectors(float* samples, float* IR, size_t ir_size, float* outputBuffer){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int samples_offset = index / ir_size;
    int ir_index = index % ir_size;
    /*
    for this part of the convolution samples start at ir_size, thats why we sum ir_size to the samples index
    for each complete multiplication we (the whole IR vector * a subvector of samples) we need to continue with a new subvector
    this new subvector is the samples vector moved one unit forward, thats why we add a samples_offset to the samples index
    */
    atomic_add(samples[ir_size + samples_offset] * IR[ir_size - 1 - ir_index], outputBuffer[ir_size + samples_offset]);
}

void convolute_toeplitz_in_gpu_kernel(float* samples, float* IR, float* outputBuffer){
    size_t ir_size = sizeof(IR) / sizeof(float);
    int threadsPerBlock = 1024;

    // first part, lower matrix multiplication
    /*
    cantidad operaciones = (ir_size * (ir_size + 1)) / 2
    esto es porque en la multiplicacion de la matriz triangular inferior
    la primera fila tiene una operacion, la segunda dos, y la n-esima (hasta el tope que es ir_size)
    equivale a Σ(n) la cual se resuelve con ((x * (x + 1)) / 2)
    */
    int blocksPerGrid;
    if (((ir_size * (ir_size + 1)) / 2) < threadsPerBlock)
        blocksPerGrid = threadsPerBlock;
    else
        blocksPerGrid = (((ir_size * (ir_size + 1)) / 2) / threadsPerBlock) + 1;
    convolute_toeplitz_lower_matrix<<<blocksPerGrid, threadsPerBlock>>>(samples, IR, ir_size, outputBuffer);
    cudaDeviceSynchronize(); // necesario¿?

    // second part, vector multiplication
    size_t samples_size = sizeof(samples) / sizeof(float);
    /*
    ya procesamos ir_size celdas del output
    quedan (samples_size - ir_size) celdas restantes
    cada celda tiene ir_size multiplicaciones
    */
    blocksPerGrid = ((samples_size - ir_size) * ir_size) / threadsPerBlock;
    convolute_toeplitz_vectors<<<blocksPerGrid, threadsPerBlock>>>(samples, IR, ir_size, outputBuffer); // todo se precisa un offset
}

__global__ void convolute_fourier(float* samples, float* IR, float* outputBuffer){
    
}

void convolute_fourier_in_gpu_kernel(float* samples, float* IR, float* outputBuffer){
    convolute_fourier(samples,IR,outputBuffer);
}

void copy_from_gpu(void* device_pointer, void* host_pointer, size_t size) {
    CUDA_CHK(cudaMemcpy(host_pointer, device_pointer, size, cudaMemcpyDeviceToHost));
};

void copy_to_gpu(void* host_pointer, void* device_pointer, size_t size) {
    CUDA_CHK(cudaMemcpy(device_pointer, host_pointer, size, cudaMemcpyHostToDevice));
};